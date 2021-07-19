import IterTools, Cairo
using Colors, Compose, Fontconfig, PyCall, StatsBase, Glob,
    FileIO, Measures, Format, Unitful, Images
pickle = pyimport("pickle")


##
@__DIR__
## https://github.com/julia-vscode/julia-vscode/issues/2104
base_dir = joinpath(@__DIR__, "final-outputs")
model_folders = filter(isdir, joinpath.(base_dir, readdir(base_dir)))
model_names = map(x->x[end],splitpath.(model_folders))
NUM_IMAGES_PER_MODEL = 10

order = ["originals", "linear_model", "64x64-mlp",
    "64x64-mlp-small", "64x64-resnet-mlp", "64x64-convnet", "perceptual_model"]

idxs = map(o->searchsortedfirst(model_names,o), order)
model_names = model_names[idxs]
model_folders = model_folders[idxs]

# load perceptual losses

file = py"""open("each_image_pl.pickle", "rb")"""
percept_losses = pickle.load(file)
avg_percept_losses = Dict(k=>mean(values(v)) for (k,v) in percept_losses)
order = sortperm(map(x->x[2],collect(avg_percept_losses)))
println("perceptual model ranking")
@show collect(avg_percept_losses)[order]

## load MSE
se(x,y) = sum((x .- y) .^ 2)

function calc_MSE(model_name; original="originals",
        data_folder=joinpath(@__DIR__,"final-outputs"))
    model_path = joinpath(data_folder, model_name)
    orig_path = joinpath(data_folder, original)
    model_imgs = glob("test*.png", model_path)
    img_names = map(x->x[end],splitpath.(model_imgs))
    cum_se = 0.0
    for name in img_names
        # x = reinterpret(UInt8,load(joinpath(model_path,name)))
        # y = reinterpret(UInt8,load(joinpath(orig_path,name)))
        x = Float64.(load(joinpath(model_path,name)))
        y = Float64.(load(joinpath(orig_path,name)))
        cum_se += se(x,y)
    end
    cum_se/length(img_names)
end


# avg_mse = Dict(k=>calc_MSE(k) for k in keys(percept_losses))
# order = sortperm(map(x->x[2],collect(avg_mse)))
# println("MSE model ranking")
# @show collect(avg_mse)[order]

# we hardcode as when reading PNGs, we don't get same loss as
# the actual floats the model trained on
avg_mse = Dict(
    "8x8-convnet" => 79.3,
    "16x16-convnet" => 52.8,
    "32x32-convnet" => 38.2,
    "64x64-convnet" => 36.6,
    "linear_model" => 42.3,
    "64x64-mlp" => 41.3,
    "64x64-mlp-small" => 38.4,
    "64x64-resnet-mlp" => 33.0,
    "32x32-convnet-descrambling" => 10.3,
    "32x32-convnet-scrambled-targets" => 72.6,
    "32x32-convnet-scrambled-inputs" => 75.2,
    "32x32-convnet-scrambled-both" => 88.8

)

##
function drawMEA(N, everyN)
    xs = collect(1:N) ./ (N+1)
    grid = hcat(collect.(IterTools.product(xs, xs))[:]...)

    grays = []
    reds = []

    for i in 1:N
        for j in 1:N
            if ((i-1) % everyN == 0) & ((j-1) % everyN == 0)
                push!(reds, [i,j])
            else
                push!(grays, [i,j])
            end
        end
    end
    
    reds = hcat(reds...) ./ (N+1)
    sz = 1 / (1.5 * N)
    
    if length(grays) >= 1
        grays = hcat(grays...) ./ (N+1)
    
        compose(context(),
                (context(), rectangle(grays[1,:], grays[2,:], [sz], [sz]), fill("darkgray")),
                (context(), rectangle(reds[1,:], reds[2,:], [sz], [sz]), fill("darkred"))
        )
    else
        compose(context(),
                (context(), rectangle(reds[1,:], reds[2,:], [sz], [sz]), fill("darkred"))
        )
    end
end

function centered_text(the_text, fs=7pt)
    compose(context(),
        text(0.5,0.5,the_text, hcenter, vcenter),
        fontsize(fs))
end

function get_images_for_model(model_folder)
    img_names = readdir(model_folder)
    # first 10 test images only
    @assert NUM_IMAGES_PER_MODEL == 10
    img_names = img_names[occursin.(r"test[0-9]-.*", img_names)]
    
    # imgs = read.(joinpath.(model_folder, img_names))
    
    imgs = load.(joinpath.(model_folder, img_names))
    imgs = map(img->adjust_histogram(img, GammaCorrection(gamma=0.8)), imgs)
    imgs = repr.("image/png",imgs)
end

## FIGURE 1 MEA SIZE
ncol = 13
nrow = 5
W = 183mm
H = W/ncol * nrow
model_names = ["$(n)x$(n)-convnet" for n in [8, 16, 32, 64]]
tab = table(nrow, ncol, 1:nrow, 1:ncol)
im_start_col = 3
tab[1,2] = [centered_text("channel\nsampling\n per 8x8")]


# draw for each # of active channels
for (i,everyN,mn) in zip(2:nrow, [8,4,2,1], model_names)
    tab[i,2] = [compose(context(), drawMEA(8, everyN))]
    model_folder = joinpath(@__DIR__, "final-outputs", mn)
    images = get_images_for_model(model_folder)
    for (idx,j) in enumerate(im_start_col:im_start_col+9)
        tab[i,j] = [compose(context(), bitmap("image/png", images[idx],0,0,1,1))]
    end
end

# original images
model_folder = joinpath(@__DIR__, "final-outputs", "originals")
images = get_images_for_model(model_folder)
# tab[6,1] = [centered_text("original")]
for (idx,j) in enumerate(im_start_col:im_start_col+9)
    tab[1,j] = [compose(context(), bitmap("image/png", images[idx],0,0,1,1))]
end

# add MSE
tab[1, ncol-1] = [centered_text("MSE")]
for (idx,mn) in zip(2:5,model_names)
    mse = "$(round(avg_mse[mn], digits=2))"
    tab[idx,ncol-1] = [centered_text(mse)]
end

# add perceptual
tab[1, ncol] = [centered_text("Percept")]
for (idx,mn) in zip(2:5,model_names)
    mse = "$(round(avg_percept_losses[mn], digits=2))"
    tab[idx,ncol] = [centered_text(mse)]
end

# add MEA cutout
circ_lw = 3pt
mea_rect = 0.6
zoom_rect = mea_rect/8
# mea = compose(context(0w, 0h, 5cm, 5cm),
mea = compose(context(),
    # zoomed selection
    # (context(), text(0.5,mea_rect/2, "64x64 channel\nHD-MEA", hcenter, vcenter),
    #     fontsize(8pt)),
    (context(), rectangle(0.5-zoom_rect/2,0.5-zoom_rect/2,zoom_rect,zoom_rect),
        stroke("red"), fill(nothing)),
    (context(), line([(1,0), (0.5+zoom_rect/2, 0.5-zoom_rect/2)]),
        strokedash([1.2mm, 1.2mm]), stroke("red")),
    (context(), line([(1,1), (0.5+zoom_rect/2, 0.5+zoom_rect/2)]),
        strokedash([1.2mm, 1.2mm]), stroke("red")),
    # MEA border
    (context(), circle(0.5cx, 0.5cy, (1cx-circ_lw)/2),
        fill(nothing), stroke("black"),linewidth(circ_lw)),
    (context(), rectangle(0.5 - mea_rect/2,0.5 - mea_rect/2,mea_rect,mea_rect),
        fill("gray80")),
)

tab[3,1] = [mea]
tab[4,1] = [ compose(context(),
        text(0.5,0.5,"64x64\nHD-MEA", hcenter),
        fontsize(7pt))]

set_default_graphic_size(W, H)
fig = compose(context(), tab)

fn = "figure1_channel_comparison"
fig |> SVG(joinpath(@__DIR__, "$fn.svg"))
@show H,W
println("$(round(H/247mm,digits=2)) of a page")
# assume 300dpi
mm2px = x -> Int(round(uconvert(u"inch",Quantity(x.value,u"mm"))*200/1u"inch",digits=0))
px_w = mm2px(W)
px_h = mm2px(H)

cmd = "inkscape -w $px_w -h $px_h $fn.svg --export-filename $fn.png"
println("to make PNG: $cmd")
fig


## FIGURE 2 model architecture
base_dir = joinpath(@__DIR__, "final-outputs")
model_folders = filter(isdir, joinpath.(base_dir, readdir(base_dir)))
model_names = map(x->x[end],splitpath.(model_folders))
NUM_IMAGES_PER_MODEL = 10

order = ["originals", "linear_model", "64x64-mlp",
    "64x64-resnet-mlp", "64x64-convnet"]

idxs = map(o->searchsortedfirst(model_names,o), order)
model_names = model_names[idxs]
model_folders = model_folders[idxs]
pretty_names = ["ground\ntruth", "linear", "MLP",
    "resMLP", "resUNet"]

ncol = 12
nrow = length(order)
W = 183mm
H = W/ncol * nrow
tab = table(nrow, ncol, 1:nrow, 1:ncol)
im_start_col = 2

# model names
for (i,mn) in enumerate(pretty_names)
    tab[i,1] = [centered_text(mn)]
end

# render images
for (i,model_folder) in zip(1:nrow, model_folders)
    images = get_images_for_model(model_folder)
    for (idx,j) in enumerate(im_start_col:im_start_col+9)
        tab[i,j] = [compose(context(), bitmap("image/png", images[idx],0,0,1,1))]
    end
end

# add MSE
tab[1, ncol-1] = [centered_text("MSE")]
for (idx,mn) in zip(2:nrow,model_names[2:end])
    mse = "$(round(avg_mse[mn], digits=2))"
    tab[idx,ncol-1] = [centered_text(mse)]
end

# add perceptual
tab[1, ncol] = [centered_text("Percept")]
for (idx,mn) in zip(2:nrow,model_names[2:end])
    mse = "$(round(avg_percept_losses[mn], digits=2))"
    tab[idx,ncol] = [centered_text(mse)]
end

set_default_graphic_size(W, H)
fig = compose(context(), tab)

fn = "figure2_model_architecture"
fig |> SVG(joinpath(@__DIR__, "$fn.svg"))
@show H,W
println("$(round(H/247mm,digits=2)) of a page")
# assume 300dpi
mm2px = x -> Int(round(uconvert(u"inch",Quantity(x.value,u"mm"))*200/1u"inch",digits=0))
px_w = mm2px(W)
px_h = mm2px(H)

cmd = "inkscape -w $px_w -h $px_h $fn.svg --export-filename $fn.png"
println("to make PNG: $cmd")
fig

## supp figure scramble
# base_dir = joinpath(@__DIR__, "final-outputs")
base_dir = "/mnt/dropbox/Dropbox/Science/manuscripts/2019_acuity_paper/2020-reconstructions/final-outputs"
model_folders = filter(isdir, joinpath.(base_dir, readdir(base_dir)))
model_names = map(x->x[end],splitpath.(model_folders))
NUM_IMAGES_PER_MODEL = 10

order = ["originals", "scrambled-targets", "32x32-convnet-descrambling", "32x32-convnet",
         "32x32-convnet-scrambled-targets", "32x32-convnet-scrambled-inputs", "32x32-convnet-scrambled-both"]

idxs = map(o->searchsortedfirst(model_names,o), order)
model_names = model_names[idxs]
model_folders = model_folders[idxs]
pretty_names = ["ground\ntruth", "scrambled", "de-\nscrambling", "baseline",
    "image\nscrambled", "retina\nscrambled", "image\n+\nretina\nscrambled"]

ncol = 12
nrow = length(order)
W = 183mm
H = W/ncol * nrow
tab = table(nrow, ncol, 1:nrow, 1:ncol)
im_start_col = 2

# model names
for (i,mn) in enumerate(pretty_names)
    tab[i,1] = [centered_text(mn)]
end

# render images
for (i,model_folder) in zip(1:nrow, model_folders)
    images = get_images_for_model(model_folder)
    for (idx,j) in enumerate(im_start_col:im_start_col+9)
        tab[i,j] = [compose(context(), bitmap("image/png", images[idx],0,0,1,1))]
    end
end

# add MSE
tab[1, ncol] = [centered_text("MSE")]
for (idx,mn) in zip(3:nrow,model_names[3:end])
    mse = "$(round(avg_mse[mn], digits=2))"
    tab[idx,ncol] = [centered_text(mse)]
end

set_default_graphic_size(W, H)
fig = compose(context(), tab)

fn = "figureSup_scrambling"
fig |> SVG(joinpath(@__DIR__, "$fn.svg"))
@show H,W
println("$(round(H/247mm,digits=2)) of a page")
# assume 300dpi
mm2px = x -> Int(round(uconvert(u"inch",Quantity(x.value,u"mm"))*200/1u"inch",digits=0))
px_w = mm2px(W)
px_h = mm2px(H)

cmd = "inkscape -w $px_w -h $px_h $fn.svg --export-filename $fn.png"
println("to make PNG: $cmd")
fig

##
fig = compose(context(textdecoration="underline"), Compose.text(0,18pt,"example text"),
            Compose.fill("black"))
