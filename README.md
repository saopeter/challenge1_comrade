# challenge1_comrade
Simple script using Comrade.jl to generate an image model for ngEHT challenge 1 data

Usage: `julia -i imaging.jl`

HMC samples are saved into the `hchain` variable. Images are centered after sampling. Plots and displays the MAP, mean, std dev, and fractional error (std dev/mean) images.
