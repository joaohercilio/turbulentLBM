"""Renderiza os quadros da simulacao (run/fields_XXXX.vtkhdf) com o ParaView.

   Isosuperficie translucida de Q, colorida por |omega_x|/|omega|-ish: o rolo (omega_x ~ 0) fica
   branco e os ribs (omega_x forte) ficam azul-ciano, imitando os tubos de bolhas da onda real.
   Entre duas saidas consecutivas os campos sao interpolados linearmente (SUB quadros por saida).
   Uso: pvbatch render.py <primeira saida> <ultima saida> [SUB] [W H]
"""

import sys, os, math
from paraview.simple import *

i0, i1 = int(sys.argv[1]), int(sys.argv[2])
SUB = int(sys.argv[3]) if len(sys.argv) > 3 else 2
W, H = (int(sys.argv[4]), int(sys.argv[5])) if len(sys.argv) > 5 else (1920, 1080)
RUN = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.environ.get("RUN", "run3"))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frames", "sim")
os.makedirs(OUT, exist_ok=True)

NX, NY, NZ, DW = 512, 512, 462, 35.124
Q_ISO = 0.4                       # em (U/delta_w)^2
DT = 0.25                         # T_REF entre saidas
N_TOTAL = 100                     # saida final do filme (t = 25), para o movimento da camera

a = VTKHDFReader(FileName=[os.path.join(RUN, "fields_%04d.vtkhdf" % i0)])
b = VTKHDFReader(FileName=[os.path.join(RUN, "fields_%04d.vtkhdf" % i0)])
mix = PythonCalculator(Input=[a, b], ArrayName="Qi",
                       Expression="inputs[0].PointData['Q']")
mixw = PythonCalculator(Input=[a, b, mix], ArrayName="wxi",
                        Expression="abs(inputs[0].PointData['omega_x'])")
merged = AppendAttributes(Input=[mix, mixw])
contour = Contour(Input=merged, ContourBy=["POINTS", "Qi"], Isosurfaces=[Q_ISO],
                  ComputeScalars=0, ComputeNormals=1, ComputeGradients=0)
contour.PointMergeMethod = "Uniform Binning"
# replica em z (periodico) para os ribs nao terminarem na borda da caixa
shift = Transform(Input=contour)
shift.Transform.Translate = [0, 0, NZ]
both = AppendDatasets(Input=[contour, shift])

view = CreateView("RenderView")
view.ViewSize = [W, H]
view.OrientationAxesVisibility = 0
view.UseColorPaletteForBackground = 0
view.BackgroundColorMode = "Gradient"
view.Background = [0.02, 0.10, 0.18]
view.Background2 = [0.20, 0.45, 0.62]
view.EnableRenderOnInteraction = 0
view.UseFXAA = 1
view.UseToneMapping = 1
view.Exposure = 1.0
view.KeyLightIntensity = 0.85
view.UseLight = 1

disp = Show(both, view)
ColorBy(disp, ("POINTS", "wxi"))
lut = GetColorTransferFunction("wxi")
lut.ApplyPreset("Blues", True)
lut.InvertTransferFunction()
lut.RescaleTransferFunction(0.0, 2.0)
lut.RGBPoints = [0.0, 0.94, 0.97, 1.0,
                 0.6, 0.75, 0.88, 0.96,
                 1.4, 0.30, 0.65, 0.88,
                 2.0, 0.10, 0.42, 0.70]
disp.Opacity = 0.55
disp.Specular = 0.6
disp.SpecularPower = 40
disp.Ambient = 0.15
disp.Diffuse = 0.85
view.EnableRenderOnInteraction = 0
view.UseColorPaletteForBackground = 0
view.DepthPeeling = 1
view.MaximumNumberOfPeels = 6

text = Text(Text="")
td = Show(text, view)
td.WindowLocation = "Lower Right Corner"
td.FontSize = 26
td.Color = [0.9, 0.95, 1.0]
td.FontFamily = "Arial"


def camera(tau):
    """Camera abaixo da camada, olhando quase ao longo do rolo (z), com leve orbita."""
    th = math.radians(-35 + 25*tau)            # azimute em torno de y
    el = math.radians(-28 + 8*tau)             # elevacao (negativa: por baixo)
    c = [0.62*NX, NY/2 + 30, 1.2*NZ]
    R = 2.1*NX
    pos = [c[0] + R*math.cos(el)*math.sin(th), c[1] + R*math.sin(el), c[2] + R*math.cos(el)*math.cos(th)]
    cam = view.GetActiveCamera()
    cam.SetFocalPoint(*c)
    cam.SetPosition(*pos)
    cam.SetViewUp(0, 1, 0)
    cam.SetViewAngle(30)


for i in range(i0, i1):
    a.FileName = [os.path.join(RUN, "fields_%04d.vtkhdf" % i)]
    b.FileName = [os.path.join(RUN, "fields_%04d.vtkhdf" % (i + 1))]
    for k in range(SUB):
        s = k/SUB
        mix.Expression = "(1 - %g)*inputs[0].PointData['Q'] + %g*inputs[1].PointData['Q']" % (s, s)
        mixw.Expression = "(1 - %g)*abs(inputs[0].PointData['omega_x']) + %g*abs(inputs[1].PointData['omega_x'])" % (s, s)
        t = (i + s)*DT
        text.Text = r"$tU/\delta_\omega = %4.1f$" % t
        camera((i + s)/N_TOTAL)
        n = i*SUB + k
        SaveScreenshot(os.path.join(OUT, "f%05d.png" % n), view, ImageResolution=[W, H])
        print("frame", n, "t =", t, flush=True)
