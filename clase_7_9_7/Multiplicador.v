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
//-----------LocalParam-----------------------//
localparam N_WORD_POT2 = 2**$clog2(N_WORDS);

/////////////////////////////////////////////////
//-----------Generate input matrix-------------//
wire signed [NB_DATA-1:0] matrix_input [N_WORDS:0] ; //contemplo los impares como pares

generate
    genvar ptr1;
    for (ptr1 = 0 ;ptr1<N_WORDS ;ptr1=ptr1+1) begin
        assign matrix_input[ptr1] = i_data[(ptr1+1)*NB_DATA-1 -: NB_DATA];
    end
endgenerate

/////////////////////////////////////////////////
//-----------Mulplication----------------------//
wire signed [(NB_DATA*2) - 1: 0] multi [(N_WORDS/2)-1:0]; //caso de impar agregar uno mas
generate
    //falta agregar condicionales
    genvar ptr2;
    if (N_WORDS%2==0)begin:Par
        for (ptr2 = 0 ;ptr2<N_WORDS ;ptr2 = ptr2+2) begin
            assign multi[ptr2-(ptr2/2)] = matrix_input[ptr2] * matrix_input[ptr2+1];
        end
    end
endgenerate

/////////////////////////////////////////////////
//-----------Parallel adder tree---------------//
//numero de etapas del sumador = clog2(N_WORDS/2)
//numero de sumas por etapa = (N_WORDS/4)/(2**n), con n = numero de etapas [0,1,2....k-1]
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
    // assign adder_vect[N_WORDS/2 - 1 - 1] = ((N_WORDS & (N_WORDS-1)) == 0) ? adder_vect[N_WORDS/2 - 1 - 1] : 
    //                                        ((N_WORDS/4)%2==0)             ? (adder_vect[N_WORDS/2 - 1 - 3] + adder_vect[N_WORDS/2 - 1 - 2]): (adder_vect[N_WORDS/4-1] + adder_vect[N_WORDS/2 - 1 - 2]) ; 
endgenerate
/////////////////////////////////////////////////
assign o_data = adder_vect[(N_WORDS/2) - 1 - 1];

endmodule