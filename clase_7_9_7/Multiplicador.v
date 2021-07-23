module Multiplicador
#(
parameter N_WORDS = 12,
parameter NB_DATA = 8
)
(
    output [NB_DATA*2 + $clog2(N_WORDS/2) - 1:0]    o_data  ,
    input  [N_WORDS*NB_DATA-1:0]                    i_data  
    // input                                           reset   ,
    // input                                           clock    
);

/////////////////////////////////////////////////
//-----------LocalParam-----------------------//
localparam N_WORD_POT2 = 2**$clog2(N_WORDS); //llevo todo el sumador interno como si fuera potencia de 2 

/////////////////////////////////////////////////
//-----------Generate input matrix-------------//
wire signed [NB_DATA-1:0] matrix_input [N_WORD_POT2:0] ; //contemplo los impares como pares

generate
    genvar ptr1;
    for (ptr1 = 0 ;ptr1<N_WORD_POT2 ;ptr1=ptr1+1) begin
        if (ptr1 < N_WORDS) begin
           assign matrix_input[ptr1] = i_data[(ptr1+1)*NB_DATA-1 -: NB_DATA];  
        end
        else begin
           assign matrix_input[ptr1]={NB_DATA{1'b0}}; 
        end
    end
endgenerate

/////////////////////////////////////////////////
//-----------Mulplication----------------------//
wire signed [(NB_DATA*2) - 1: 0] multi [(N_WORD_POT2/2):0]; //contemplo los impares como pares
generate
    //falta agregar condicionales
    genvar ptr2;
    if (N_WORDS%2==0)begin:ParProducto
        for (ptr2 = 0 ;ptr2<N_WORD_POT2 ;ptr2 = ptr2+2) begin
            assign multi[ptr2-(ptr2/2)] = matrix_input[ptr2] * matrix_input[ptr2+1];
        end
    end
    else begin:ImparesProducto
        for (ptr2 = 0 ;ptr2<N_WORD_POT2-1 ;ptr2 = ptr2+2) begin
            assign multi[ptr2-(ptr2/2)] = matrix_input[ptr2] * matrix_input[ptr2+1];
        end
        assign multi[N_WORD_POT2/2] = {{NB_DATA{matrix_input[N_WORDS-1][NB_DATA-1]}},matrix_input[N_WORDS-1]}; //ya tiene expancion
    end
endgenerate

/////////////////////////////////////////////////
//-----------Parallel adder tree---------------//
//numero de etapas del sumador = clog2(N_WORDS/2)
//numero de sumas por etapa = (N_WORDS/4)/(2**n), con n = numero de etapas [0,1,2....k-1]
wire signed [NB_DATA*2 + $clog2(N_WORDS/2) - 1:0] adder_vect        [(N_WORD_POT2/2) - 1 : 0]; //necesito (N_words/2) - 1 sumadores en caso de N_words potencia de 2
wire signed [NB_DATA*2 + $clog2(N_WORDS/2) - 1:0] adder_vect_aux    [(N_WORD_POT2/2) - 1 : 0]; //vector auxiliar para la expancion del signo
wire signed [NB_DATA*2 + $clog2(N_WORDS/2) - 1:0] o_data_aux ; // para guardar la ultima suma
generate
    genvar ptr3,ptr4;
    if(N_WORDS%2==0)begin:ParSuma
        for (ptr4 = 0;ptr4<$clog2(N_WORD_POT2/2) ;ptr4=ptr4+1) begin
            for (ptr3 = 0 ;ptr3 < (N_WORD_POT2/4)/(2**ptr4) ;ptr3 = ptr3+1 ) begin //N_WORDS/2 por aumentar el ptr3 + 2, sino seria N_WORDS/4 y ptr3 + 1 y las seleccion seria con 2**ptr3 
                if (ptr4==0) begin
                    assign adder_vect_aux[ptr3] = multi[2*ptr3] + multi[(2*ptr3)+1];
                    assign adder_vect[ptr3] = {{$clog2(N_WORDS/2){adder_vect_aux[ptr3][NB_DATA*2-1]}},adder_vect_aux[ptr3][NB_DATA*2-1 -: NB_DATA*2]};
                end
                else begin
                    assign adder_vect_aux[ptr3 + N_WORD_POT2/2 - ((N_WORD_POT2/2)/(2**ptr4))] = adder_vect[2*ptr3 + N_WORD_POT2/2 - (N_WORD_POT2/(2**ptr4))] + adder_vect[2*ptr3 + N_WORD_POT2/2 - (N_WORD_POT2/(2**ptr4)) + 1];
                    assign adder_vect[ptr3 + N_WORD_POT2/2 - ((N_WORD_POT2/2)/(2**ptr4))] = {{$clog2(N_WORDS/2)-ptr4{adder_vect_aux[ptr3 + N_WORD_POT2/2 - ((N_WORD_POT2/2)/(2**ptr4))][NB_DATA*2 + ptr4 -1]}},adder_vect_aux[ptr3 + N_WORD_POT2/2 - ((N_WORD_POT2/2)/(2**ptr4))][NB_DATA*2 + ptr4 -1 -: NB_DATA*2 + ptr4]};
                end
            end 
        end
        assign o_data_aux = adder_vect[(N_WORD_POT2/2) - 1 - 1];
    end
    else begin
        for (ptr4 = 0;ptr4<$clog2(N_WORD_POT2/2) ;ptr4=ptr4+1) begin
            for (ptr3 = 0 ;ptr3 < (N_WORD_POT2/4)/(2**ptr4) ;ptr3 = ptr3+1 ) begin //N_WORDS/2 por aumentar el ptr3 + 2, sino seria N_WORDS/4 y ptr3 + 1 y las seleccion seria con 2**ptr3 
                if (ptr4==0) begin
                    assign adder_vect[ptr3] = multi[2*ptr3] + multi[(2*ptr3)+1]; 
                end
                else begin
                    assign adder_vect[ptr3 + N_WORD_POT2/2 - ((N_WORD_POT2/2)/(2**ptr4))] = adder_vect[2*ptr3 + N_WORD_POT2/2 - (N_WORD_POT2/(2**ptr4))] + adder_vect[2*ptr3 + N_WORD_POT2/2 - (N_WORD_POT2/(2**ptr4)) + 1];
                end
            end 
        end
        assign o_data_aux = multi[N_WORD_POT2/2] + adder_vect[(N_WORD_POT2/2) - 1 - 1];
    end
    
endgenerate
/////////////////////////////////////////////////
assign o_data = o_data_aux;

endmodule