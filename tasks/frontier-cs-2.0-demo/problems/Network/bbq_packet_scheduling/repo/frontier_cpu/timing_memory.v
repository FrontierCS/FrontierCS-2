// Timing boundaries of the supplied RAM/FIFO adapters. Every input is captured
// by the clock and every output is launched by the clock; the adapters contain
// no input-to-output combinational path. Authored for this task.
(* blackbox *)
module bram_simple2port #(
    parameter DWIDTH = 16, AWIDTH = 9, DEPTH = 512, IS_OUTDATA_REG = 1
) (
    input clock, input [DWIDTH-1:0] data,
    input [AWIDTH-1:0] rdaddress, input rden,
    input [AWIDTH-1:0] wraddress, input wren,
    output wire [DWIDTH-1:0] q
);
endmodule

(* blackbox *)
module sc_fifo #(
    parameter DWIDTH = 8, DEPTH = 2048, IS_SHOWAHEAD = 0, IS_OUTDATA_REG = 0
) (
    input clock, input [DWIDTH-1:0] data, input rdreq, wrreq,
    output wire empty, full, output wire [DWIDTH-1:0] q,
    output wire [$clog2(DEPTH)-1:0] usedw
);
endmodule
