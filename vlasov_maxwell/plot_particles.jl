
using GLVisualize, GeometryTypes, Colors
using Colors
w = glscreen(color = RGBA(0f0, 0f0, 0f0, 0f0));
@async renderloop(w)


Np=10000;
particles=rand(Float32,3,Np);



cat    = GLNormalMesh(loadasset("cat.obj"))
sphere = GLNormalMesh(Sphere{Float32}(Vec3f0(0), 1f0), 12)




# color        = foldp(color_gen, colorstart, t)
rotation     = -sphere.normals
ps           = sphere.vertices

ps=GeometryTypes.Point3f0.( particles[1,:], particles[2,:],particles[3,:])
rotation=GeometryTypes.Normal.( particles[3,:], particles[2,:],particles[1,:])
# cat=GeometryTypes.Circle(GeometryTypes.Point2f0(0), 0.2f0)
cat=GeometryTypes.Sphere{Float32}(Vec3f0(0), 0.01f0)
# color = fill(RGBA{Float32}(1., 1., 1., 0.9), size(particles,2))
color = fill(RGBA{Float32}(1., 1., 1., 0.05), size(particles,2))

cats = visualize((cat, ps),boundingbox=nothing,color=color, billboard = true)
obj=cats.children[]
_view(cats, w)

# obj[:color][1: Int(Np/2)].=RGBA{Float32}(1., 1., 0., 0.9)


# GeometryTypes.Point3f0.(rand(Float32,3,4)))
#
# visualize(
#    (GeometryTypes.Circle(GeometryTypes.Point2f0(0), 0.002f0),
#       GeometryTypes.Point3f0.(particles)))
#
# particle_vis = visualize(
#     (GeometryTypes.Circle(GeometryTypes.Point2f0(0), 0.002f0),
#        GeometryTypes.Point3f0.(particles)),
#     boundingbox = nothing, # don't waste time on bb computation
#     color = fill(RGBA{Float32}(0, 0, 0, 0.09), length(particles)),
#     billboard = true
# ).children[]
# _view(particle_vis, camera = :perspective)
#
# particle_vis[:color][1:n_particles] = map(1:n_particles) do i
#     xx = (i / n_particles) * 2pi
#     RGBA{Float32}((sin(xx) + 1) / 2, (cos(xx) + 1.0) / 2.0, 0.0, 0.1)
# end
# color = fill(RGBA{Float32}(0, 0, 0, 0.09), length(particles))
