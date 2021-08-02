

module ejemplosGenerate
    #(
        parameter NB_DATA = 8,
        parameter N_WORD  = 8
    )
    (
        output [ NB_DATA         - 1 : 0] o_data ,
        input  [(NB_DATA*N_WORD) - 1 : 0] i_data ,
        input                             reset  , 
        input                             clock
    );

    /////////////////////////////////////////////
    // Localparam: Se cambian localmente
    /////////////////////////////////////////////
    localparam NB_TOTAL = NB_DATA*N_WORD;

    /////////////////////////////////////////////
    // Vars
    // wire: representa logica combinacional o cables
    // reg:
    // - LC: condicionada a todas las variable '*'
    // - LS: condicionada al posedge clock o negedge clock 
    /////////////////////////////////////////////
    
    wire signed [7 :0] cte1;
    wire signed [7 :0] cte2;
    wire signed [15:0] producto;
    wire signed [8 :0] suma;
    wire signed [8 :0] suma1;

    assign producto = cte1 * cte2;
    assign suma     = cte1 + cte2;

    /////////////////////////////////////////////
    wire signed [7:0] cp_data0;
    wire signed [7:0] cp_data1;
    assign cp_data0 = i_data[7 :0];
    assign cp_data1 = i_data[15:8];
    assign suma1    = cp_data0 + cp_data1;

    assign suma1    = $signed(i_data[7:0]) + $signed(i_data[15:8]);

    /////////////////////////////////////////////
    reg signed [7:0] matrix [7:0];
    integer ptr;

    always @(*) begin:labelAlways
        for(ptr=0;ptr<N_WORD;ptr=ptr+1) begin:labelEx
            matrix[ptr] = i_data[(ptr+1)*NB_DATA - 1 -: NB_DATA];
        end
    end
    assign suma1 = matrix[0] + matrix[1];

    //----------------------------------//

    wire signed [7:0] matrix_w [7:0];   
    generate
        genvar ptr1;
        for(ptr1=0;ptr1<N_WORD;ptr1=ptr1+1) begin:labelEx1
            assign matrix_w[ptr] = i_data[(ptr+1)*NB_DATA - 1 -: NB_DATA];
        end
    endgenerate
    assign suma1 = matrix[0] + matrix[1];
    /////////////////////////////////////////////

    wire signed [7:0] matrix_w [7:0];   

    generate
        genvar ptr1;

        if(N_WORD%2==0) begin
            for(ptr1=0;ptr1<N_WORD;ptr1=ptr1+1) begin:labelEx1
                assign matrix_w[ptr] = i_data[(ptr+1)*NB_DATA - 1 -: NB_DATA];
            end
        end
        else begin
            for(ptr1=0;ptr1<N_WORD;ptr1=ptr1+1) begin:labelEx1
                assign matrix_w[ptr] = i_data[(ptr+1)*NB_DATA - 1 -: NB_DATA];
            end          
            assign matrix_w[7] = 0;
        end
    endgenerate

    assign suma1 = matrix[0] + matrix[1];

    /////////////////////////////////////////////

    reg signed [7:0] matrix_w [7:0];   

    generate
        genvar ptr1;

        if(N_WORD%2==0) begin
            for(ptr1=0;ptr1<N_WORD;ptr1=ptr1+1) begin:labelEx1
                always @(*) begin
                    matrix_w[ptr] = i_data[(ptr+1)*NB_DATA - 1 -: NB_DATA];                 
                end
            end
        end
        else begin
            for(ptr1=0;ptr1<N_WORD;ptr1=ptr1+1) begin:labelEx1
                always @(*) begin
                    matrix_w[ptr] = i_data[(ptr+1)*NB_DATA - 1 -: NB_DATA];
                end            
            end          
            always @(*)
                matrix_w[7] = 0;
        end
    endgenerate

    assign suma1 = matrix[0] + matrix[1];

    /////////////////////////////////////////////

    reg signed [7:0] matrix_w [7:0];   

    generate
        genvar ptr1;

        if(N_WORD%2==0) begin
            for(ptr1=0;ptr1<N_WORD;ptr1=ptr1+1) begin:labelEx1
                always @(posedge clock) begin
                    matrix_w[ptr] <= i_data[(ptr+1)*NB_DATA - 1 -: NB_DATA];                 
                end
            end
        end
        else begin
            for(ptr1=0;ptr1<N_WORD;ptr1=ptr1+1) begin:labelEx1
                always @(posedge clock) begin
                    matrix_w[ptr] <= i_data[(ptr+1)*NB_DATA - 1 -: NB_DATA];
                end            
            end          
            always @(posedge clock)
                matrix_w[7] <= 0;
        end
    endgenerate

    assign suma1 = matrix[0] + matrix[1];


    /////////////////////////////////////////////

    localparam N_INST = 8;
    wire [(N_INST*NB_DATA)     - 1 : 0] dataSamples0;
    wire [(N_INST*NB_DATA)     - 1 : 0] dataSamples1;
    wire [(N_INST*(NB_DATA+1)) - 1 : 0] dataAdderOut;

    generate
        genvar ptr2;

        for(ptr2=0;ptr2<N_INST;ptr2=ptr2+1) begin:exInst

            fullAdder
                u_fullAdder
                (
                    .outData (dataAdderOut[(ptr2+1)*(NB_DATA+1) - 1 -: NB_DATA+1]),
                    .inData0 (dataSamples0[(ptr2+1)*(NB_DATA  ) - 1 -: NB_DATA  ]),
                    .inData1 (dataSamples1[(ptr2+1)*(NB_DATA  ) - 1 -: NB_DATA  ]),
                    .clock   (clock                                              )
                )

        end


    endgenerate

    /////////////////////////////////////////////
    localparam MODO = 0;
    generate
        genvar ptr2;

        if(MODO==0) begin
            for(ptr2=0;ptr2<N_INST;ptr2=ptr2+1) begin:exInst
                fullAdder
                    u_fullAdder
                    (
                        .outData (dataAdderOut[(ptr2+1)*(NB_DATA+1) - 1 -: NB_DATA+1]),
                        .inData0 (dataSamples0[(ptr2+1)*(NB_DATA  ) - 1 -: NB_DATA  ]),
                        .inData1 (dataSamples1[(ptr2+1)*(NB_DATA  ) - 1 -: NB_DATA  ]),
                        .clock   (clock                                              )
                    )
            end
        end
        else begin
            for(ptr2=0;ptr2<N_INST;ptr2=ptr2+1) begin:exInst
                fullAdderModo
                    u_fullAdderModo
                    (
                        .outData (dataAdderOut[(ptr2+1)*(NB_DATA+1) - 1 -: NB_DATA+1]),
                        .inData0 (dataSamples0[(ptr2+1)*(NB_DATA  ) - 1 -: NB_DATA  ]),
                        .inData1 (dataSamples1[(ptr2+1)*(NB_DATA  ) - 1 -: NB_DATA  ]),
                        .clock   (clock                                              )
                    )
            end
        end

    endgenerate

    /////////////////////////////////////////////
    `define MODO 0;
    
    `ifdef MODO
                fullAdder
                    u_fullAdder
                    (
                        .outData (dataAdderOut[(NB_DATA+1) - 1 -: NB_DATA+1]),
                        .inData0 (dataSamples0[(NB_DATA  ) - 1 -: NB_DATA  ]),
                        .inData1 (dataSamples1[(NB_DATA  ) - 1 -: NB_DATA  ]),
                        .clock   (clock                                    )
                    )
        
    `else
                fullAdderModo
                    u_fullAdderModo
                    (
                        .outData (dataAdderOut[(NB_DATA+1) - 1 -: NB_DATA+1]),
                        .inData0 (dataSamples0[(NB_DATA  ) - 1 -: NB_DATA  ]),
                        .inData1 (dataSamples1[(NB_DATA  ) - 1 -: NB_DATA  ]),
                        .clock   (clock                                     )
                    )
    `endif

    /////////////////////////////////////////////

    // Option 1
    wire signed [7:0] matrix_w [N_WORD-1:0];   
    wire        [(N_WORD/2)*16 - 1 : 0] oData;
    generate
        genvar ptr1;

        for(ptr1=0;ptr1<N_WORD;ptr1=ptr1+2) begin:labelEx1
            assign matrix_w[ptr1  ] = i_data[(ptr1+1)*NB_DATA - 1 -: NB_DATA];
            assign matrix_w[ptr1+1] = i_data[(ptr1+2)*NB_DATA - 1 -: NB_DATA];
            assign oData[(ptr1-ptr1/2+1)*16 - 1 -: 16] = matrix_w[ptr1] * matrix_w[ptr1+1];
        end

    endgenerate

    // Option 2
    wire        [(N_WORD/2)*16 - 1 : 0] oData;
    generate
        genvar ptr1;

        for(ptr1=0;ptr1<N_WORD;ptr1=ptr1+2) begin:labelEx1
            assign oData[(ptr1-ptr1/2+1)*16 - 1 -: 16] = $signed(i_data[(ptr1+1)*NB_DATA - 1 -: NB_DATA]) * $signed(i_data[(ptr1+2)*NB_DATA - 1 -: NB_DATA]);
        end

    endgenerate

    // Option 3
    wire        [(N_WORD/2)*16 - 1 : 0] oData;
    generate
        genvar ptr1;

        wire signed [7:0] tmp0;
        wire signed [7:0] tmp1;

        for(ptr1=0;ptr1<N_WORD;ptr1=ptr1+2) begin:labelEx1
            assign tmp0 = i_data[(ptr1+1)*NB_DATA - 1 -: NB_DATA];
            assign tmp1 = i_data[(ptr1+2)*NB_DATA - 1 -: NB_DATA];
            assign oData[(ptr1-ptr1/2+1)*16 - 1 -: 16] = tmp0 * tmp1;
        end

    endgenerate

    /////////////////////////////////////////////


    /////////////////////////////////////////////
    // Ejemplo de instancia de modulo con referencias de localparam
    /////////////////////////////////////////////
    // fir
    //     (
    //         .NB_DATA(NB_TOTAL),
    //         .N_WORD (10)
    //      )
    //     fir0();


endmodule

/////////////////////////////////////////////
// Ejemplode construcion de parametros y puertos
/////////////////////////////////////////////
// module ejemplosGenerate
//     ( o_data ,i_data,reset ,clock);

//     parameter NB_DATA = 10;
//     parameter N_WORD  = 8;

//     output [ NB_DATA         - 1 : 0] o_data ;
//     input  [(NB_DATA*N_WORD) - 1 : 0] i_data ;
//     input                             reset  ; 
//     input                             clock  ;


// endmodule
