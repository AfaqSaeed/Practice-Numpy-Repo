from pattern import Checker,Circle,Spectrum
from generator import ImageGenerator

c = Circle(512, 20, (50, 50))
c.draw()
c.show()

check = Checker(250, 25)
check.draw()
check.show()

spec = Spectrum(255)
spec.draw()
spec.show()

