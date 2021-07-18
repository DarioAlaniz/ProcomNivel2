module Multiplicador
#(
parameter N_WORDS = 32,
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
wire signed [NB_DATA*2 + $clog2(N_WORDS) - 1:0] adder_vect [(N_WORDS/2) - 1 - 1 : 0];//necesito (N_words/2) - 1 sumadores en caso de N_words potencia de 2
generate
    genvar ptr3,ptr4;
    for (ptr4 = 0;ptr4<$clog2(N_WORDS/2) ;ptr4=ptr4+1) begin
        for (ptr3 = 0 ;ptr3 < (N_WORDS/4)/(2**ptr4) ;ptr3 = ptr3+1 ) begin //N_WORDS/2 por aumentar el ptr3 + 2, sino seria N_WORDS/4 y ptr3 + 1 y las seleccion seria con 2**ptr3 
            if (ptr4==0) begin
                assign adder_vect[ptr3] = multi[2*ptr3] + multi[(2*ptr3)+1]; 
            end
            else begin
                assign adder_vect[ptr3 + N_WORDS/2 - ((N_WORDS/2)/(2**ptr4))] = adder_vect[2*ptr3 + N_WORDS/2 - (N_WORDS/(2**ptr4))] + adder_vect[2*ptr3 + N_WORDS/2 - (N_WORDS/(2**ptr4)) + 1];
            end
        end 
    end

endgenerate
/////////////////////////////////////////////////
assign o_data = adder_vect[(N_WORDS/2) - 1 - 1];

endmodule