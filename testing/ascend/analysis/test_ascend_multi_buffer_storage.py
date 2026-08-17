import tilelang.language as T
import tvm
from tilelang.engine.lower import lower
from tvm import tirx
from tvm.tirx.stmt_functor import post_order_visit


def _make_alias_claim_program():
    size = 16

    @T.prim_func
    def main(A: T.Buffer((size,), "float32"), C: T.Buffer((size,), "float32")):
        with T.Kernel(1):
            ub = T.alloc_shared((size,), "float32")
            alias = T.reshape(ub, (4, 4))
            T.annotate_buffer_versions({ub: 2})

            # Deliberately list two Buffer aliases. The annotation pass must
            # normalize this frontend form to one storage Var.
            for i in T.Pipelined(
                size,
                num_stages=2,
                annotations={"multi_buffer_eligible": [alias, ub]},
            ):
                ub[i] = A[i]
                C[i] = alias[i // 4, i % 4] + T.float32(1)

    return main


def _collect_unique_buffers(stmt, storage):
    buffers = []

    def add(buffer):
        if not buffer.data.same_as(storage):
            return
        if not any(buffer.same_as(existing) for existing in buffers):
            buffers.append(buffer)

    def visit(node):
        if isinstance(node, (tirx.BufferLoad, tirx.BufferStore)):
            add(node.buffer)

    post_order_visit(stmt, visit)
    return buffers


def test_multi_buffer_contract_uses_storage_vars_for_aliases():
    snapshots = {}
    wanted = {
        "tl.AnnotateMultiBufferEligible",
        "tl.AutoSchedule",
        "tl.InsertSync",
    }

    @tvm.ir.instrument.pass_instrument
    class CapturePassIR:
        def run_after_pass(self, mod, info):
            if info.name in wanted:
                snapshots[info.name] = mod

    with tvm.transform.PassContext(opt_level=3, instruments=[CapturePassIR()]):
        lower(_make_alias_claim_program(), target="ascend")

    claims = []

    def collect_claims(node):
        if not isinstance(node, tirx.For):
            return
        claim = node.annotations.get("multi_buffer_eligible")
        if claim:
            claims.append(claim)

    annotated = snapshots["tl.AnnotateMultiBufferEligible"]["main"]
    post_order_visit(annotated.body, collect_claims)
    assert len(claims) == 1
    assert len(claims[0]) == 1
    storage = claims[0][0]
    assert isinstance(storage, tirx.Var)

    selected_versions = []

    def collect_selected_versions(node):
        if not isinstance(node, tirx.SBlock):
            return
        versions = node.annotations.get("tl.buffer_versions_map")
        if versions:
            selected_versions.append(versions)

    scheduled = snapshots["tl.AutoSchedule"]["main"]
    post_order_visit(scheduled.body, collect_selected_versions)
    assert len(selected_versions) == 1
    assert len(selected_versions[0]) == 1
    selected_storage = next(iter(selected_versions[0]))
    assert selected_storage.same_as(storage)
    assert int(selected_versions[0][selected_storage]) == 2

    synchronized = snapshots["tl.InsertSync"]["main"]
    versioned_buffers = _collect_unique_buffers(synchronized.body, storage)
    versioned_shapes = {tuple(int(dim) for dim in buffer.shape) for buffer in versioned_buffers}
    assert versioned_shapes == {(2, 16), (2, 4, 4)}


if __name__ == "__main__":
    test_multi_buffer_contract_uses_storage_vars_for_aliases()
