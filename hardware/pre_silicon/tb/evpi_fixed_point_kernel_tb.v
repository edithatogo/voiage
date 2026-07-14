`timescale 1ns/1ps

module evpi_fixed_point_kernel_tb;
    reg  [31:0] expected_with_information;
    reg  [31:0] expected_without_information;
    wire [31:0] evpi_value;

    evpi_fixed_point_kernel dut (
        .expected_with_information(expected_with_information),
        .expected_without_information(expected_without_information),
        .evpi_value(evpi_value)
    );

    task check_case;
        input [31:0] with_information;
        input [31:0] without_information;
        input [31:0] expected;
        begin
            expected_with_information = with_information;
            expected_without_information = without_information;
            #1;
            if (evpi_value !== expected) begin
                $display("EVPI fixed-point mismatch: with=%0d without=%0d expected=%0d got=%0d",
                         with_information, without_information, expected, evpi_value);
                $finish;
            end
        end
    endtask

    initial begin
        check_case(32'd1250, 32'd1000, 32'd250);
        check_case(32'd900,  32'd1000, 32'd0);
        check_case(32'd1000, 32'd1000, 32'd0);
        check_case(32'd8745, 32'd6120, 32'd2625);
        $display("evpi_fixed_point_kernel_tb passed");
        $finish;
    end
endmodule
