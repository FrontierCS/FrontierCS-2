// Task-owned port observer. No queue implementation is contained here.
`timescale 1ns/10ps
import heap_ops::*;
module trace_tb;
parameter D=2, W=4, N=63;
localparam PW=$clog2(W**D);
logic clk=0;
always #5 clk=~clk;
logic rst=0, ready, iv=0, ov;
heap_op_t op=HEAP_OP_ENQUE, oop;
logic [63:0] data=0, odata;
logic [PW-1:0] priority_in=0, priority_out;
bbq #(.HEAP_BITMAP_WIDTH(W),.HEAP_ENTRY_DWIDTH(64),.HEAP_MAX_NUM_ENTRIES(N)) dut(
 .clk(clk),.rst(rst),.ready(ready),.in_valid(iv),.in_op_type(op),
 .in_he_data(data),.in_he_priority(priority_in),.out_valid(ov),
 .out_op_type(oop),.out_he_data(odata),.out_he_priority(priority_out));
integer inputs, events, rc, cycle=0, nr, nv, no, np;
longint unsigned nd;
string input_path, event_path;
initial begin
 if (!$value$plusargs("INPUT=%s",input_path)) $fatal;
 if (!$value$plusargs("EVENTS=%s",event_path)) $fatal;
 inputs=$fopen(input_path,"r"); events=$fopen(event_path,"w");
 if (!inputs || !events) $fatal;
 while (!$feof(inputs)) begin
  @(negedge clk);
  rc=$fscanf(inputs,"%d %d %d %d %d\n",nr,nv,no,nd,np);
  if(rc!=5) $fatal;
  rst=nr; iv=nv; op=heap_op_t'(no); data=nd; priority_in=PW'(np);
  @(posedge clk);
  // Acceptance is sampled before NBA. Completion is sampled after NBA.
  if(iv && !rst) $fwrite(events,"A %0d %0d\n",cycle,ready);
  #1;
  if(ov) $fwrite(events,"O %0d %0d %0d %0d\n",cycle,oop,odata,priority_out);
  cycle=cycle+1;
 end
 $fwrite(events,"DONE %0d\n",cycle);
 $fclose(events); $fclose(inputs); $finish;
end
initial begin #10000000; $fatal(1,"trusted observation deadline"); end
endmodule
