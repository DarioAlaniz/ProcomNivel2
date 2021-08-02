

module generateData2
  #(parameter NB_DATA = 8,
    parameter ROWS    = 8
    )
   (output [NB_DATA*ROWS - 1 : 0] o_data,
    input [NB_DATA*ROWS  - 1 : 0] i_data,
    input 		 	  clock			 
    );

   generate
      genvar 			  ptr;

      for(ptr=0;ptr<ROWS;ptr=ptr+1)begin: forconnect

	 fulladder
	      #(.NB_DATA(NB_DATA))
	 u_fulladder
	      (.o_data   (o_data[(ptr+1)*NB_DATA - 1 -: NB_DATA]),
	       .i_data   (i_data[(ptr+1)*NB_DATA - 1 -: NB_DATA]),
	       .clock    (clock)
	       );
      end
   endgenerate
   
	 


endmodule // generateData2
