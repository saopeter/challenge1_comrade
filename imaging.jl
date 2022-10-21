using Pkg; Pkg.activate(@__DIR__)
Pkg.add(url="https://github.com/ptiede/RadioImagePriors.jl")
using Comrade
using Distributions
using ComradeOptimization
using ComradeAHMC
using OptimizationBBO
using Plots
using StatsBase
using OptimizationOptimJL
using RadioImagePriors
using DistributionsAD

# load eht-imaging we use this to load eht data
load_ehtim()
# To download the data visit https://doi.org/10.25739/g85n-f134
obs = ehtim.obsdata.load_uvfits(joinpath(@__DIR__, "synthetic_data/M87_eht2022_230_thnoise.uvfits"))
obs.add_scans()
# kill 0-baselines since we don't care about
# large scale flux and make scan-average data
obs = scan_average(obs).add_fractional_noise(0.01)
# extract log closure amplitudes and closure phases
damp = extract_amp(obs)
dcphase = extract_cphase(obs)
lklhd = RadioLikelihood(damp, dcphase)

# Build the Model. Here we we a struct to hold some caches
# This will be useful to hold precomputed caches

struct GModel{C,G}
    cache::C
    gcache::G
    fovx::Float64
    fovy::Float64
    nx::Int
    ny::Int
    function GModel(obs::Comrade.EHTObservation, fovx, fovy, nx, ny)
        buffer = IntensityMap(zeros(ny, nx), fovx, fovy, BSplinePulse{3}())
        # cache = create_cache(DFTAlg(obs), buffer)
        cache = create_cache(NFFTAlg(obs), buffer)
        gcache = GainCache(scantable(obs))
        return new{typeof(cache), typeof(gcache)}(cache, gcache, fovx, fovy, nx, ny)
    end
end

function (model::GModel)(θ)
    # (;c, f, fg, lgamp) = θ
    (;c, f, fg) = θ
    # Construct the image model
    img = IntensityMap(f*c, model.fovx, model.fovy, BSplinePulse{3}())
    m = modelimage(img, model.cache)
    gaussian = fg*stretched(Gaussian(), μas2rad(1000.0), μas2rad(1000.0))
    # Now corrupt the model with Gains
    # g = exp.(lgamp)
    return m+gaussian
    # Comrade.GainModel(model.gcache, g, m+gaussian)
end


# Define the station gain priors
# the stations we're using are PV, KP, SM, AA, PB, AZ, GL, JC, AP, LM
# add KP, PB, GL
distamp = (AA = Normal(0.0, 0.1),
           AP = Normal(0.0, 0.1),
        #    LM = Normal(-0.5, 0.9),
           LM = Normal(0.0, 0.1),
           AZ = Normal(0.0, 0.1),
           JC = Normal(0.0, 0.1),
           PV = Normal(0.0, 0.1),
           SM = Normal(0.0, 0.1),
           KP = Normal(0.0, 0.1),
           PB = Normal(0.0, 0.1),
           GL = Normal(0.0, 0.1)
           )


fovx = μas2rad(200.0) #originally 75
fovy = μas2rad(200.0)
nx = 32
ny = floor(Int, fovy/fovx*nx)

# imgfoo = IntensityMap(zeros(ny,nx), fovx, fovy)
# # xitr, yitr = imagepixels(mms1(xopt).model)
# xitr, yitr = imagepixels(imgfoo)
# prior = (
#     c = CenteredImage(xitr, yitr, 1.0, ImageDirichlet(1.0, ny, nx)),
#     f = Uniform(0.0, 1.0),
#     fg = Uniform(0.0, 1.0),
#     #σG = Uniform(μas2rad(200.0), μas2rad(2000.0)),
#     #lgamp1 = Comrade.GainPrior(distamp, scantable(damp1)),
#     )
    
xitr, yitr = Comrade.imagepixels(fovx, fovy, nx, ny)
prior = (
              c = ImageDirichlet(1.0, ny, nx),
              f = Uniform(0.2, 0.9),
              fg = Uniform(0.0, 1.0),
            #   lgamp = Comrade.GainPrior(distamp, scantable(damp)),
            )
        
        
mms = GModel(damp, fovx, fovy, nx, ny)

post = Posterior(lklhd, prior, mms)

tpost = asflat(post)

# We will use HMC to sample the posterior.

# Now lets zoom to the peak using LBFGS
ndim = dimension(tpost)
using Zygote
f = OptimizationFunction(tpost, Optimization.AutoZygote())
prob = OptimizationProblem(f, rand(ndim) .- 0.5, nothing)
ℓ = logdensityof(tpost)
sol = solve(prob, LBFGS(); maxiters=10000, callback=(x,p)->(@info ℓ(x); false), g_tol=1e-1)

xopt = transform(tpost, sol)

# Let's see how the fit looks
residual(mms(xopt), damp)
residual(mms(xopt), dcphase)
plot(mms(xopt), fovx=fovx, fovy=fovy, title="MAP", colorbar_scale=:log10, clims=(1e-6, 1e-4))

# Let's also plot the gain curves
# gt = Comrade.caltable(mms(xopt))
# plot(gt, ylims=:none, layout=(4,3), size=(600,500))

using Measurements


# now we sample using hmc
metric = DiagEuclideanMetric(ndim)
hchain, stats = sample(post, AHMC(;metric, autodiff=AD.ZygoteBackend()), 5000; nadapts=4000, init_params=xopt)

# # Now plot the gain table with error bars
# gamps = exp.(hcat(hchain.lgamp...))
# mga = mean(gamps, dims=2)
# sga = std(gamps, dims=2)

# using Measurements
# gmeas = measurement.(mga, sga)
# ctable = caltable(mms.gcache, vec(gmeas))
# plot(ctable)

# Plot the mean image and standard deviation image
using StatsBase
samples = mms.(sample(hchain, 50))
imgs = intensitymap.(samples, fovx, fovy, 96, 96)

shifted_imgs = map(similar, imgs)
for i in eachindex(shifted_imgs)
    xc = centroid(samples[i].m1.image)
    smodel = shifted(samples[i].m1, -xc[1], -xc[2])
    simg = intensitymap(smodel, imgs[i].fovx, imgs[i].fovy, size(imgs[i])...)
    shifted_imgs[i] .= simg .+ 1e-6 # add a floor to prevent zero flux
end

mimg, simg = mean_and_std(shifted_imgs)

p1 = plot(mimg, title="Mean", clims=(1e-4, maximum(mimg)), dpi=500, colorbar_scale=:log10)
display(p1)
savefig("mean_centered.png")

p2 = plot(simg,  title="Std. Dev.", clims=(1e-4, maximum(mimg)), dpi=500, colorbar_scale=:log10)
display(p2)
savefig("std_centered.png")

p3 = plot(simg./mimg,  title="Fractional Error", xlims=(-60,60), ylims=(-60,60), dpi=500, colorbar_scale=:log10)
display(p3)
savefig("error_centered_cropped.png")


# Computing information 
# ```
# Julia Version 1.8.1
# Commit afb6c60d69a (2022-09-06 15:09 UTC)
# Platform Info:
#   OS: Linux (x86_64-linux-gnu)
#   CPU: 16 × Intel(R) Xeon(R) CPU @ 2.20GHz
#   WORD_SIZE: 64
#   LIBM: libopenlibm
#   LLVM: libLLVM-13.0.1 (ORCJIT, broadwell)
#   Threads: 16 on 16 virtual cores
# Environment:
#   LD_LIBRARY_PATH = /usr/local/cuda/lib64:/usr/local/nccl2/lib:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/lib64:/usr/local/nccl2/lib:/usr/local/cuda/extras/CUPTI/lib64
#   JULIA_NUM_THREADS = 16
# ```