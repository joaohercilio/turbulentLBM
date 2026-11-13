import os, sys, time
import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

N, WMAX = 512, 5.0
OUT = sys.argv[1] if len(sys.argv) > 1 else "/home/geoenergialab/turbulentLBM/hit_omega.mp4"
cmap = plt.get_cmap("inferno")

img = vtk.vtkImageData()
img.SetDimensions(N, N, N)
fe = vtk.vtkFlyingEdges3D()
fe.SetInputData(img)
fe.SetValue(0, 0.52*255)
fe.ComputeNormalsOn()
fe.InterpolateAttributesOn()
fe.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, "Omega")
lut = vtk.vtkLookupTable()
lut.SetNumberOfTableValues(256)
for k in range(256):
    lut.SetTableValue(k, *cmap(k/255))
lut.SetRange(0, 255)
mapper = vtk.vtkPolyDataMapper()
mapper.SetInputConnection(fe.GetOutputPort())
mapper.SetScalarModeToUsePointFieldData()
mapper.SelectColorArray("w")
mapper.SetLookupTable(lut)
mapper.SetScalarRange(0, 0.6*255)
mapper.ScalarVisibilityOn()
actor = vtk.vtkActor()
actor.SetMapper(mapper)
actor.GetProperty().SetSpecular(0.3); actor.GetProperty().SetAmbient(0.25)
actor.GetProperty().SetSpecularPower(20)
outline = vtk.vtkOutlineFilter()
outline.SetInputData(img)
om = vtk.vtkPolyDataMapper()
om.SetInputConnection(outline.GetOutputPort())
oa = vtk.vtkActor()
oa.SetMapper(om)
oa.GetProperty().SetColor(0.7, 0.7, 0.7)
ren = vtk.vtkRenderer()
ren.AddActor(actor)
ren.AddActor(oa)
ren.SetBackground(0, 0, 0)
vtk.vtkLightKit().AddLightsToRenderer(ren)
win = vtk.vtkRenderWindow()
win.SetOffScreenRendering(1)
win.AddRenderer(ren)
win.SetSize(1080, 1080)
cam = ren.GetActiveCamera()
cam.ParallelProjectionOff()
cam.SetViewAngle(25)
c = (N - 1)/2
cam.SetFocalPoint(c, c, c)
cam.SetPosition(c + 2.2*N, c - 2.9*N, c + 1.8*N)
cam.SetViewUp(0, 0, 1)
grab = vtk.vtkWindowToImageFilter()
grab.SetInput(win)

def iso(path, first):
    q = np.fromfile(path, dtype=np.uint8)
    o, w = q[:N**3], q[N**3:]
    a, b = numpy_to_vtk(o, deep=True), numpy_to_vtk(w, deep=True)
    a.SetName("Omega"); b.SetName("w")
    pd = img.GetPointData()
    for name in ("Omega", "w"):
        pd.RemoveArray(name)
    pd.AddArray(a); pd.AddArray(b)
    pd.SetActiveScalars("Omega")
    img.Modified()
    if first:
        ren.ResetCamera(); cam.Zoom(0.9)
    win.Render()
    grab.Modified(); grab.Update()
    im = grab.GetOutput()
    nx, ny, _ = im.GetDimensions()
    pic = vtk_to_numpy(im.GetPointData().GetScalars()).reshape(ny, nx, -1)[::-1]
    return pic, w[:N*N].reshape(N, N).astype(float)*WMAX/255

plt.style.use("dark_background")
plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix"})
fig = plt.figure(figsize=(12.8, 7.2))
a1 = fig.add_axes([0.0, 0.0, 0.5625, 1.0]); a1.set_axis_off()
D = os.path.dirname(os.path.abspath(__file__)) + "/.."
zs = np.load(f"{D}/spectral/realization1/spectrum.npz")
zl = np.load(f"{D}/lbm/d3q27_3rd/Ma_x1/realization1/spectrum.npz")
k = zs["kc"][1:int(zs["N"])//3]
sel = np.unique(np.geomspace(1, len(k), 26).astype(int)) - 1
spectrum = lambda z, t: np.exp([np.interp(t, z["t_te"], np.log(z["E"][:, j + 1])) for j in range(len(k))])

a2 = fig.add_axes([0.63, 0.2, 0.33, 0.62])
p1 = a1.imshow(np.zeros((1080, 1080, 3), np.uint8))
ps, = a2.loglog(k, spectrum(zs, 0), color="white", lw=1.8, label="Pseudo-Spectral")
lb, = a2.loglog(k[sel], spectrum(zl, 0)[sel], "o", color="#ff9f1c", mfc="none", ms=6, mew=1.4, label="LBM")
a2.set_xlim(1, k[-1]); a2.set_ylim(1e-13, 3e-4)
a2.set_xlabel(r"$\kappa$", fontsize=15); a2.set_ylabel(r"$E(\kappa)$", fontsize=15)
a2.tick_params(labelsize=12, direction="in", which="both", top=True, right=True)
a2.legend(loc="lower left", frameon=False, fontsize=14)
a2.set_title("Energy spectrum", fontsize=16)
fig.text(0.02, 0.945, "Decaying isotropic turbulence", fontsize=20)
fig.text(0.02, 0.905, r"3rd-order regularization, $512^3$, $Re_\lambda \approx 120$", fontsize=13, color="0.75")
fig.text(0.02, 0.04, r"$\Omega$ criterion (Liu et al. 2016) colored by vorticity", fontsize=13, color="0.75")
clock = fig.text(0.63, 0.06, "", fontsize=18)

alive = lambda: any(open(f"/proc/{p}/comm").read().strip() == "hit" for p in os.listdir("/proc") if p.isdigit() and os.path.exists(f"/proc/{p}/comm"))
wr = FFMpegWriter(fps=24, bitrate=20000)
with wr.saving(fig, OUT, dpi=150):
    i = 0
    while True:
        f = f"frames/{i}.bin"
        while not os.path.exists(f) and alive():
            time.sleep(1)
        if not os.path.exists(f):
            break
        pic, sl = iso(f, i == 0)
        tt = dict(np.loadtxt("times.dat", ndmin=2))[i]
        p1.set_data(pic)
        ps.set_ydata(spectrum(zs, tt)); lb.set_ydata(spectrum(zl, min(tt, zl["t_te"][-1]))[sel])
        clock.set_text(rf"$t/t_e = {tt:.2f}$")
        wr.grab_frame()
        os.remove(f)
        print(i, flush=True)
        i += 1
