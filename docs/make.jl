using Documenter
using VlasovPIF

makedocs(
    sitename = "VlasovPIF",
    format = Documenter.HTML(),
    modules = [VlasovPIF]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#