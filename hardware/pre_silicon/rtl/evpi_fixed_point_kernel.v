// Fixed-point EVPI-style kernel for pre-silicon evidence.
//
// Inputs use a shared integer scale recorded in the fixture bundle. The result
// is max(expected_with_information - expected_without_information, 0).

module evpi_fixed_point_kernel #(
    parameter WIDTH = 32
) (
    input  wire [WIDTH-1:0] expected_with_information,
    input  wire [WIDTH-1:0] expected_without_information,
    output wire [WIDTH-1:0] evpi_value
);
    assign evpi_value = expected_with_information > expected_without_information
        ? expected_with_information - expected_without_information
        : {WIDTH{1'b0}};
endmodule
