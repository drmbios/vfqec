OPENQASM 3.0;
include "stdgates.inc";
gate isometry_to_uncompute_dg _gate_q_0, _gate_q_1, _gate_q_2 {
}
gate state_preparation(_gate_p_0, _gate_p_1, _gate_p_2, _gate_p_3, _gate_p_4, _gate_p_5, _gate_p_6, _gate_p_7) _gate_q_0, _gate_q_1, _gate_q_2 {
  isometry_to_uncompute_dg _gate_q_0, _gate_q_1, _gate_q_2;
}
bit[2] c;
qubit[5] q;
state_preparation(1.0, 0, 0, 0, 0, 0, 0, 0) q[0], q[1], q[2];
cx q[0], q[3];
cx q[1], q[3];
c[0] = measure q[3];
cx q[1], q[4];
cx q[2], q[4];
c[1] = measure q[4];
