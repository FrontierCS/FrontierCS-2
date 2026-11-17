// Diagnostic CPU simulation adapters; authored locally, not upstream Intel IP.
// This experiment covers only the parameter modes used by generated BBQ RTL.
`timescale 1 ns/10 ps
module bram_simple2port #(
    parameter DWIDTH = 16, AWIDTH = 9, DEPTH = 512, IS_OUTDATA_REG = 1
) (
    input clock, input [DWIDTH-1:0] data,
    input [AWIDTH-1:0] rdaddress, input rden,
    input [AWIDTH-1:0] wraddress, input wren,
    output wire [DWIDTH-1:0] q
);
    reg [DWIDTH-1:0] mem [0:DEPTH-1];
    reg [DWIDTH-1:0] read_data = 0, output_data = 0;
    initial begin
        for (integer k = 0; k < DEPTH; k = k + 1) mem[k] = 0;
    end
    always @(posedge clock) begin
        if (wren) mem[wraddress] <= data;
        // Intel's mixed-port collision mode is DONT_CARE. Verilator is
        // two-state, so separate collision-value experiments are still needed.
        if (rden) read_data <= (wren && wraddress == rdaddress) ? 'x : mem[rdaddress];
        output_data <= read_data;
    end
    assign q = IS_OUTDATA_REG ? output_data : read_data;
endmodule

module sc_fifo #(
    parameter DWIDTH = 8, DEPTH = 2048, IS_SHOWAHEAD = 0, IS_OUTDATA_REG = 0
) (
    input clock, input [DWIDTH-1:0] data, input rdreq, wrreq,
    output wire empty, full, output reg [DWIDTH-1:0] q = 0,
    output wire [$clog2(DEPTH)-1:0] usedw
);
    reg [DWIDTH-1:0] mem [0:DEPTH-1];
    integer head = 0, tail = 0, count = 0;
    wire take = rdreq && !empty;
    wire put = wrreq && !full;
    initial begin
        if (IS_SHOWAHEAD != 0 || IS_OUTDATA_REG != 1)
            $fatal(1, "Unsupported diagnostic FIFO mode");
    end
    assign empty = count == 0;
    assign full = count == DEPTH;
    assign usedw = count;
    always @(posedge clock) begin
        if (take) begin
            q <= mem[head];
            head <= head == DEPTH-1 ? 0 : head+1;
        end
        if (put) begin
            mem[tail] <= data;
            tail <= tail == DEPTH-1 ? 0 : tail+1;
        end
        case ({put, take})
            2'b10: count <= count+1;
            2'b01: count <= count-1;
            default: ;
        endcase
    end
endmodule
