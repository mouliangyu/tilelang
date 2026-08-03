#include "./latency_estimator.h"

#include <tvm/runtime/logging.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>

#include <algorithm>
#include <limits>
#include <unordered_map>

#include "op/builtin.h"
#include "transform/common/attr.h"

namespace tvm {
namespace tl {

using namespace tirx;

namespace {

struct PerCoreTaskCost {
  int64_t latency{0};
  int64_t ii{0};
};

int64_t CheckedAdd(int64_t lhs, int64_t rhs, const char *field) {
  ICHECK_GE(lhs, 0) << "PerCoreTask " << field << " must be non-negative";
  ICHECK_GE(rhs, 0) << "PerCoreTask " << field << " must be non-negative";
  ICHECK_LE(lhs, std::numeric_limits<int64_t>::max() - rhs)
      << "PerCoreTask " << field << " overflow while aggregating phases";
  return lhs + rhs;
}

int64_t CheckedMultiply(int64_t value, int64_t factor, const char *field) {
  ICHECK_GE(value, 0) << "PerCoreTask " << field << " must be non-negative";
  ICHECK_GE(factor, 0) << "PerCoreTask trip count must be non-negative";
  if (value == 0 || factor == 0)
    return 0;
  ICHECK_LE(value, std::numeric_limits<int64_t>::max() / factor)
      << "PerCoreTask " << field << " overflow while multiplying by trip count "
      << factor;
  return value * factor;
}

PerCoreTaskCost MaxCost(const PerCoreTaskCost &lhs,
                        const PerCoreTaskCost &rhs) {
  return {std::max(lhs.latency, rhs.latency), std::max(lhs.ii, rhs.ii)};
}

PerCoreTaskCost AddCost(const PerCoreTaskCost &lhs,
                        const PerCoreTaskCost &rhs) {
  return {CheckedAdd(lhs.latency, rhs.latency, "latency"),
          CheckedAdd(lhs.ii, rhs.ii, "II")};
}

PerCoreTaskCost MultiplyCost(const PerCoreTaskCost &cost, int64_t factor) {
  return {CheckedMultiply(cost.latency, factor, "latency"),
          CheckedMultiply(cost.ii, factor, "II")};
}

struct PhaseSummary {
  bool has_boundary{false};
  PerCoreTaskCost prefix;
  PerCoreTaskCost middle;
  PerCoreTaskCost suffix;
  size_t candidate_markers{0};
  size_t boundary_markers{0};
};

void AppendSummary(PhaseSummary *dst, const PhaseSummary &src) {
  dst->candidate_markers += src.candidate_markers;
  dst->boundary_markers += src.boundary_markers;
  if (!src.has_boundary) {
    if (dst->has_boundary) {
      dst->suffix = MaxCost(dst->suffix, src.prefix);
    } else {
      dst->prefix = MaxCost(dst->prefix, src.prefix);
    }
    return;
  }

  if (!dst->has_boundary) {
    dst->has_boundary = true;
    dst->prefix = MaxCost(dst->prefix, src.prefix);
    dst->middle = src.middle;
    dst->suffix = src.suffix;
    return;
  }

  dst->middle = AddCost(dst->middle, MaxCost(dst->suffix, src.prefix));
  dst->middle = AddCost(dst->middle, src.middle);
  dst->suffix = src.suffix;
}

PhaseSummary RepeatSummary(const PhaseSummary &body, int64_t trip_count) {
  if (trip_count <= 0) {
    PhaseSummary result;
    // Marker counts describe the static statement tree, not the number of
    // dynamic executions.  Preserve them even when the loop has no work so
    // validation does not misreport a missing Task or wait marker.
    result.candidate_markers = body.candidate_markers;
    result.boundary_markers = body.boundary_markers;
    return result;
  }
  PhaseSummary result = body;
  if (trip_count == 1)
    return result;

  if (!body.has_boundary) {
    // Without an inter-core wait there is no proof that different loop
    // iterations are mutually exclusive across cores. Keep the serial-loop
    // estimate conservative.
    result.prefix = MultiplyCost(body.prefix, trip_count);
    return result;
  }

  // One physical core executes its suffix from iteration i before its prefix
  // from iteration i + 1.  They are serial parts of the same compound task;
  // only distinct Task candidates within one phase may overlap and
  // use MaxCost.
  PerCoreTaskCost between_iterations = AddCost(body.suffix, body.prefix);
  result.middle = AddCost(MultiplyCost(body.middle, trip_count),
                          MultiplyCost(between_iterations, trip_count - 1));
  return result;
}

using CandidateCostMap =
    std::unordered_map<Stmt, PerCoreTaskCost, ObjectPtrHash, ObjectPtrEqual>;

class PhaseAnalyzer {
public:
  struct Result {
    PerCoreTaskCost cost;
    bool has_boundary{false};
  };

  static Result Analyze(const Stmt &stmt,
                        const CandidateCostMap &candidate_costs,
                        const ConstrSet &outer_ctx) {
    PhaseAnalyzer analyzer(candidate_costs);
    outer_ctx.Populate(analyzer.arith_analyzer_);
    PhaseSummary summary = analyzer.AnalyzeStmt(stmt);
    ICHECK_EQ(summary.candidate_markers, candidate_costs.size())
        << "Failed to match every T.Task marker while estimating "
           "PerCoreTask latency";
    ICHECK_EQ(summary.boundary_markers, analyzer.CountWaitMarkers(stmt))
        << "Failed to match every inter-core wait while estimating PerCoreTask "
           "latency";

    PerCoreTaskCost cost = summary.prefix;
    if (summary.has_boundary) {
      cost = AddCost(cost, summary.middle);
      cost = AddCost(cost, summary.suffix);
    }
    return {cost, summary.has_boundary};
  }

private:
  explicit PhaseAnalyzer(const CandidateCostMap &candidate_costs)
      : candidate_costs_(candidate_costs) {}

  static bool IsInterCoreWait(const EvaluateNode *evaluate) {
    const auto *call = evaluate->value.as<CallNode>();
    return call && call->op.same_as(tl::ascend_cross_core_wait_flag());
  }

  size_t CountWaitMarkers(const Stmt &stmt) const {
    class Counter : public StmtExprVisitor {
    public:
      size_t count{0};

    private:
      void VisitExpr_(const CallNode *op) final {
        if (op->op.same_as(tl::ascend_cross_core_wait_flag()))
          ++count;
        StmtExprVisitor::VisitExpr_(op);
      }
    } counter;
    counter(stmt);
    return counter.count;
  }

  PhaseSummary AnalyzeStmt(const Stmt &stmt) const {
    if (!stmt.defined())
      return {};
    if (const auto *seq = stmt.as<SeqStmtNode>()) {
      PhaseSummary result;
      for (const Stmt &child : seq->seq)
        AppendSummary(&result, AnalyzeStmt(child));
      return result;
    }
    if (const auto *attr = stmt.as<AttrStmtNode>()) {
      if (attr->attr_key == tl::attr::kAscendTask) {
        Stmt marker = GetRef<Stmt>(attr);
        auto it = candidate_costs_.find(marker);
        ICHECK(it != candidate_costs_.end())
            << "Missing latency estimate for T.Task marker";
        PhaseSummary result;
        result.prefix = it->second;
        result.candidate_markers = 1;
        return result;
      }
      return AnalyzeStmt(attr->body);
    }
    if (const auto *evaluate = stmt.as<EvaluateNode>()) {
      if (IsInterCoreWait(evaluate)) {
        PhaseSummary result;
        result.has_boundary = true;
        result.boundary_markers = 1;
        return result;
      }
      return {};
    }
    if (const auto *loop = stmt.as<ForNode>()) {
      ICHECK(loop->kind == ForKind::kSerial || loop->kind == ForKind::kUnrolled)
          << "Control flow selecting T.Task candidates must use a "
             "serial or unrolled loop";
      return RepeatSummary(
          AnalyzeStmt(loop->body),
          GetSerialTripCount(GetRef<For>(loop), &arith_analyzer_));
    }
    if (const auto *while_loop = stmt.as<WhileNode>()) {
      // Match ControlNode::GetTripCount's dynamic-loop fallback.
      return RepeatSummary(AnalyzeStmt(while_loop->body),
                           kDynamicSerialTripCountFallback);
    }
    if (const auto *branch = stmt.as<IfThenElseNode>()) {
      PhaseSummary then_summary = AnalyzeStmt(branch->then_case);
      PhaseSummary else_summary = branch->else_case
                                      ? AnalyzeStmt(branch->else_case.value())
                                      : PhaseSummary{};
      ICHECK(!then_summary.has_boundary && !else_summary.has_boundary)
          << "Inter-core synchronization inside T.PerCoreTask must be outside "
             "conditional control flow";
      PhaseSummary result;
      result.prefix = MaxCost(then_summary.prefix, else_summary.prefix);
      result.candidate_markers =
          then_summary.candidate_markers + else_summary.candidate_markers;
      result.boundary_markers =
          then_summary.boundary_markers + else_summary.boundary_markers;
      return result;
    }
    if (const auto *block = stmt.as<SBlockNode>()) {
      PhaseSummary result;
      if (block->init)
        AppendSummary(&result, AnalyzeStmt(block->init.value()));
      AppendSummary(&result, AnalyzeStmt(block->body));
      return result;
    }
    if (const auto *realize = stmt.as<SBlockRealizeNode>())
      return AnalyzeStmt(realize->block);
    return {};
  }

  const CandidateCostMap &candidate_costs_;
  mutable arith::Analyzer arith_analyzer_;
};

bool IsComposedLatencyStmt(const Stmt &stmt) {
  if (stmt.as<SeqStmtNode>() || stmt.as<AttrStmtNode>() ||
      stmt.as<IfThenElseNode>() || stmt.as<WhileNode>()) {
    return true;
  }
  if (const auto *loop = stmt.as<ForNode>()) {
    return loop->kind == ForKind::kSerial || loop->kind == ForKind::kUnrolled;
  }
  return false;
}

} // namespace

void LatencyEstimator::EstimateComposedTask(
    TaskNode *task, const LeafTaskAnalyzer &analyze_leaf) {
  if (!IsComposedLatencyStmt(task->stmt)) {
    Estimate(task);
    return;
  }
  SerialTaskCost cost =
      EstimateStmtCost(task->stmt, task->outer_ctx, analyze_leaf);
  task->SetLatency(cost.latency);
  task->SetII(std::max<int64_t>(cost.ii, 1));
}

LatencyEstimator::SerialTaskCost
LatencyEstimator::EstimateLeafTaskCost(const Stmt &stmt,
                                       const ConstrSet &outer_ctx,
                                       const LeafTaskAnalyzer &analyze_leaf) {
  TaskNode leaf_task;
  leaf_task.stmt = stmt;
  leaf_task.outer_ctx = outer_ctx;
  analyze_leaf(stmt, &leaf_task);
  Estimate(&leaf_task);
  return {leaf_task.GetLatency(), leaf_task.GetII()};
}

LatencyEstimator::SerialTaskCost
LatencyEstimator::EstimateStmtCost(const Stmt &stmt, const ConstrSet &outer_ctx,
                                   const LeafTaskAnalyzer &analyze_leaf) {
  if (!stmt.defined())
    return {};

  auto add_cost = [&](const SerialTaskCost &lhs, const SerialTaskCost &rhs) {
    if (lhs.ii == 0)
      return rhs;
    if (rhs.ii == 0)
      return lhs;
    return SerialTaskCost{
        std::max(lhs.latency,
                 CheckedSerialTaskAdd(lhs.ii, rhs.latency, "latency")),
        CheckedSerialTaskAdd(lhs.ii, rhs.ii, "II")};
  };
  auto multiply_cost = [&](const SerialTaskCost &cost, int64_t factor) {
    if (factor <= 0 || cost.ii == 0)
      return SerialTaskCost{};
    return SerialTaskCost{
        CheckedSerialTaskAdd(
            CheckedSerialTaskMultiply(cost.ii, factor - 1, "latency"),
            cost.latency, "latency"),
        CheckedSerialTaskMultiply(cost.ii, factor, "II")};
  };

  if (const auto *seq = stmt.as<SeqStmtNode>()) {
    SerialTaskCost result;
    ConstrSet child_ctx = outer_ctx;
    for (const Stmt &child : seq->seq) {
      result =
          add_cost(result, EstimateStmtCost(child, child_ctx, analyze_leaf));
      if (const auto *bind = child.as<BindNode>())
        child_ctx.AddConstr(bind->var, bind->value);
    }
    return result;
  }
  if (const auto *attr = stmt.as<AttrStmtNode>()) {
    ConstrSet body_ctx = outer_ctx;
    if (attr->attr_key == tirx::attr::tilelang_assume) {
      if (const auto *expr = attr->node.as<PrimExprNode>())
        body_ctx.AddConstr(GetRef<PrimExpr>(expr), /*is_assume=*/true);
    }
    return EstimateStmtCost(attr->body, body_ctx, analyze_leaf);
  }
  if (const auto *loop = stmt.as<ForNode>()) {
    if (loop->kind != ForKind::kSerial && loop->kind != ForKind::kUnrolled)
      return EstimateLeafTaskCost(stmt, outer_ctx, analyze_leaf);

    arith::Analyzer analyzer;
    outer_ctx.Populate(analyzer);
    int64_t trip_count = GetSerialTripCount(GetRef<For>(loop), &analyzer);
    ConstrSet body_ctx = outer_ctx;
    body_ctx.AddConstr(loop->loop_var,
                       Range::FromMinExtent(loop->min, loop->extent));
    body_ctx.AddConstr(loop->extent > 0);
    return multiply_cost(EstimateStmtCost(loop->body, body_ctx, analyze_leaf),
                         trip_count);
  }
  if (const auto *while_loop = stmt.as<WhileNode>()) {
    ConstrSet body_ctx = outer_ctx;
    body_ctx.AddConstr(while_loop->condition);
    return multiply_cost(
        EstimateStmtCost(while_loop->body, body_ctx, analyze_leaf),
        kDynamicSerialTripCountFallback);
  }
  if (const auto *branch = stmt.as<IfThenElseNode>()) {
    ConstrSet then_ctx = outer_ctx;
    then_ctx.AddConstr(branch->condition);
    SerialTaskCost then_cost =
        EstimateStmtCost(branch->then_case, then_ctx, analyze_leaf);
    SerialTaskCost else_cost;
    if (branch->else_case) {
      ConstrSet else_ctx = outer_ctx;
      else_ctx.AddConstr(Not(branch->condition));
      else_cost =
          EstimateStmtCost(branch->else_case.value(), else_ctx, analyze_leaf);
    }
    return {std::max(then_cost.latency, else_cost.latency),
            std::max(then_cost.ii, else_cost.ii)};
  }
  return EstimateLeafTaskCost(stmt, outer_ctx, analyze_leaf);
}

void LatencyEstimator::EstimatePerCoreTask(
    TaskNode *per_core_task, const Stmt &body,
    const std::vector<PerCoreTaskCandidate> &candidates) {
  CandidateCostMap candidate_costs;
  for (const PerCoreTaskCandidate &candidate : candidates) {
    ICHECK(candidate.task);
    ICHECK(!candidate.task->GetReadRegions().empty() ||
           !candidate.task->GetWriteRegions().empty())
        << "T.Task must contain at least one dependency-bearing "
           "statement";
    bool inserted = candidate_costs
                        .emplace(candidate.marker,
                                 PerCoreTaskCost{candidate.task->GetLatency(),
                                                 candidate.task->GetII()})
                        .second;
    ICHECK(inserted) << "Duplicate T.Task marker identity";
  }

  PhaseAnalyzer::Result phase_cost =
      PhaseAnalyzer::Analyze(body, candidate_costs, per_core_task->outer_ctx);
  per_core_task->SetLatency(phase_cost.cost.latency);
  // Mode-0 inter-core flags are explicit, unversioned protocol state. Keep
  // dynamic protocol invocations non-overlapping until a versioned flag
  // contract is introduced.
  per_core_task->SetII(std::max<int64_t>(
      phase_cost.has_boundary ? phase_cost.cost.latency : phase_cost.cost.ii,
      1));
}

} // namespace tl
} // namespace tvm
