module Multiplicador
#(
parameter N_WORDS = 16,
parameter NB_DATA = 8
)
(
    output [NB_DATA*2 + $clog2(N_WORDS) - 1:0]  o_data  ,
    input  [N_WORDS*NB_DATA-1:0]                i_data  ,
    input                                       reset   ,
    input                                       clock    
);

/////////////////////////////////////////////////
//-----------Generate input matrix-------------//
wire signed [NB_DATA-1:0] matrix_input [N_WORDS-1:0] ;

generate
    genvar ptr1;
    for (ptr1 = 0 ;ptr1<N_WORDS ;ptr1=ptr1+1) begin
        assign matrix_input[ptr1] = i_data[(ptr1+1)*NB_DATA-1 -: NB_DATA];
    end
endgenerate

/////////////////////////////////////////////////
//-----------Mulplication----------------------//
wire signed [(NB_DATA*2) - 1: 0] multi [(N_WORDS/2)-1:0];
generate
    //falta agregar condicionales
    genvar ptr2;
    for (ptr2 = 0 ;ptr2<N_WORDS ;ptr2 = ptr2+2) begin:multi_pot_2
        assign multi[ptr2-(ptr2/2)] = matrix_input[ptr2] * matrix_input[ptr2+1];
    end
endgenerate

/////////////////////////////////////////////////
//-----------Parallel adder tree---------------//
wire signed [NB_DATA*2 + $clog2(N_WORDS) - 1:0] adder; 
wire signed [NB_DATA*2 + $clog2(N_WORDS) - 1:0] adder_vect [(N_WORDS/2) - 1 : 0];//necesito (N_words/2) - 1 sumadores en caso de N_words potencia de 2
generate
    genvar ptr3;
    for (ptr3 = 0 ;ptr3 < N_WORDS/2 ;ptr3 = ptr3+2 ) begin
        assign adder_vect[ptr3-(ptr3/2)] = multi[ptr3] + multi[ptr3+1];
    end
    assign adder = adder_vect[0] + adder_vect[1] + adder_vect[2] + adder_vect[3];  

endgenerate
/////////////////////////////////////////////////
assign o_data = adder;

endmodule