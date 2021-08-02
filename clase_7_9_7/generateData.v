

module generateData
  #(parameter DATA_TYPE = 0
   )
  (output [7      : 0] o_data,
   input [8*8 - 1 : 0] i_data,  // Vectors
   input 	       clock
   );

   
   //###############################
   //## Always
   //###############################
   wire signed [7       : 0] vardataW [7:0];
   reg signed [ 7       : 0] varDataR [7:0]; // [8b] name [8row] 
   
   integer 	       ptrf;
   
   always@(*)begin
      for(ptrf=0;ptrf<N_BLOCKS;ptrf=ptrf+1)begin:foralways
	 varDataR[ptrf] = i_data[(ptrf+)*8-1 -: 8];  // varData[row][b] = Input 8*8
      end
   end
   
   wire [NB_DATA - 1 : 0] dataFor [N_LENGTH_DATA - 1 : 0];
   
   // LLamada de instancias
   generate
      genvar ptr;

      if(N_LENGTH_DATA%2==0)begin: par
	 for(ptr=0;ptr<N_LENGTH_DATA;ptr=ptr+1):begin:forData
	    assign dataFor[ptr] = i_data[(ptrf+)*8-1 -: 8];
	 end
      end
      else begin
	 for(ptr=0;ptr<N_LENGTH_DATA-1;ptr=ptr+1):begin:forData
	    assign dataFor[ptr] = i_data[(ptrf+)*8-1 -: 8];
	 end
	 assign dataFor[N_LENGTH_DATA - 1] = i_data[8*8 - 1 -: 8];
      end // else: !if(N_LENGTH_DATA%2==0)
   
   endgenerate

      // LLamada de instancias
   generate
      genvar ptr;

      if(N_LENGTH_DATA%2==0)begin: par
	 always@(posedge clock) begin
	    for(ptr=0;ptr<N_LENGTH_DATA;ptr=ptr+1):begin:forData
	       dataFor[ptr] <= i_data[(ptrf+)*8-1 -: 8];
	    end
	 end
      end
      else begin
	 always@(posedge clock) begin
	    for(ptr=0;ptr<N_LENGTH_DATA-1;ptr=ptr+1):begin:forData
	       dataFor[ptr] <= i_data[(ptrf+)*8-1 -: 8];
	    end
	    dataFor[N_LENGTH_DATA - 1] <= i_data[8*8 - 1 -: 8];
	 end
      end // else: !if(N_LENGTH_DATA%2==0)
      
   endgenerate


   generate
      if(DATA_TYPE == 0)begin
	 fulladder
	   u_fulladder();
      end
      else begin
	fulladderMode
	  u_fulladderMode();
      end
   endgenerate

   //wire [15:0] tmpwire [N_LENGTH_DATA - 1 : 0];
   generate
      genvar ptr;
      wire [15:0] tmpwire;
      
      for(ptr=0;ptr<N_LENGTH_DATA;ptr=ptr+1):begin:forData
	 //assign tmpwire[ptr]              = i_data[(ptr+1)*8 -1: 8] * i_data2[(ptr+1)*8 -1 : 8];
	 //assign o_data[(ptr+1)*16 -1: 16] = tmpwire[ptr];
	 assign tmpwire                   = i_data[(ptr+1)*8 -1: 8] * i_data2[(ptr+1)*8 -1 : 8];
	 assign o_data[(ptr+1)*16 -1: 16] = tmpwire;
      end

      for(ptr=0;ptr<N_LENGTH_DATA;ptr=ptr+1):begin:forData
	 fulladder
	  u_fulldadder();
      end
      
   endgenerate

   ///////////////////////////
   ///////////////////////////
   //`define PAR_DATA
   
   `ifdef PAR_DATA
   always@(*)begin
      for(ptrf=0;ptrf<N_BLOCKS;ptrf=ptrf+1)begin:foralways
	 varDataR[ptrf] = i_data[(ptrf+)*8-1 -: 8];  // varData[row][b] = Input 8*8
      end
   end
   `else
   always@(*)begin
      for(ptrf=0;ptrf<N_BLOCKS-1;ptrf=ptrf+1)begin:foralways
	 varDataR[ptrf] = i_data[(ptrf+)*8-1 -: 8];  // varData[row][b] = Input 8*8
      end
      varDataR[N_BLOCKS-1] = i_data[8*8-1 -: 8];
   end
   `endif

      

   

endmodule // generateData
