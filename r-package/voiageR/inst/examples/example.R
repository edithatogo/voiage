# Example usage of voiageR

# Load the package
library(voiageR)

# Check if voiage is available
if (!is_voiage_available()) {
  stop("voiage Python package is not available. Please install it with: pip install voiage")
}

# Create sample data
set.seed(123)

# Net benefit data: 1000 PSA samples, 2 strategies
net_benefits <- matrix(rnorm(2000), nrow = 1000, ncol = 2)
colnames(net_benefits) <- c("Strategy A", "Strategy B")

# Calculate EVPI
cat("Calculating EVPI...\n")
evpi_result <- evpi(net_benefits)
cat("EVPI:", evpi_result, "\n")

# Parameter samples for EVPPI
param_samples <- list(
  efficacy = rnorm(1000, mean = 0.7, sd = 0.1),
  cost = rnorm(1000, mean = 1000, sd = 200)
)

# Calculate EVPPI
cat("Calculating EVPPI...\n")
evppi_result <- evppi(net_benefits, param_samples)
cat("EVPPI:", evppi_result, "\n")

# Population-scaled EVPI
cat("Calculating population-scaled EVPI...\n")
evpi_pop <- evpi(
  net_benefits = net_benefits,
  population = 100000,
  time_horizon = 10,
  discount_rate = 0.03
)
cat("Population EVPI:", evpi_pop, "\n")

cat("Example completed successfully!\n")