#!/usr/bin/env python

# raytrace.py -- Seth Just

from math import floor
from numpy import sqrt, pi, sin, cos, e
import numpy
import Image

EPSILON = 1e-8

class Coord:    
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __add__(self, other):
        if not isinstance(other, Vector):
            raise TypeError
        return Coord(self.x+other.x, self.y+other.y, self.z+other.z)
            
    def __sub__(self, other):
        return Vector(self.x-other.x, self.y-other.y, self.z-other.z)

    def __repr__(self):
        return "(%s, %s, %s)"%(str(self.x), str(self.y), str(self.z))

    def __getitem__(self, i):
        if i == 0: return self.x
        elif i == 1: return self.y
        elif i == 2: return self.z
        else: raise IndexError

    def dist(self, other):
        if isinstance(other, Vector):
            raise TypeError
        return abs(self-other)

class Vector(Coord):
    def __init__(self, x, y, z):
        Coord.__init__(self, x, y, z)
        self.R = None

    def __add__(self, other):
      if not isinstance(other, Coord):
        raise TypeError
      return Vector(self.x+other.x, self.y+other.y, self.z+other.z)
        
    def __mul__(self, other):
        return self.x*other.x + self.y*other.y + self.z*other.z

    def __abs__(self):
        return sqrt(sum(map(lambda x: x**2, list(self))))

    def norm(self):
        return self.scale(1/abs(self))

    def scale(self, s):
        return Vector(self.x * s, self.y * s, self.z * s)

    def cross(self, other):
        return Vector(self.y * other.z - self.z * other.y, self.z * other.x - self.x * other.z, self.x * other.y - self.y * other.x)

    def rotate(self, V):
        # returns the vector V passed through a transformation rotating (0,0,-1) to self.norm()

        if self.R == None:
            t = self.norm().scale(-1)
            u = Vector(1,0,0)
            if u*t == 1: u = Vector(0,1,0)
            u = (u - t.scale(u*t)).norm()
            v = t.cross(u).norm()
            
            r = [u, v, t]

            self.R = [Vector(r[0][i], r[1][i], r[2][i]) for i in range(3)]

        r = map(lambda r: r*V, self.R)
        return Vector(r[0], r[1], r[2])

class Color:
    def __init__(self, r, g, b, a=0):
        self.r = int(r)
        self.g = int(g)
        self.b = int(b)
        self.a = 0

    def __getitem__(self, i):
        if i == 0:
            return self.r
        elif i == 1:
            return self.g
        elif i == 2:
            return self.b
        elif i == 3:
            return self.a
        else:
            raise IndexError

    def __add__(self, other):
        return Color(self.r+other.r, self.g+other.g, self.b+other.b)

    def __mul__(self, a):
        if isinstance(a, Color):
            return Color(self.r*a.r, self.g*a.g, self.b*a.b)/255
        return Color(self.r*a, self.g*a, self.b*a)

    def __pow__(self, a):
        t = map(lambda x: ((x/255)**a)/255, list(self))
        return Color(t[0], t[1], t[2])

    def __div__(self, a):
        return self*(1/a)

class SkyColor(Color):
  def __init__(self):
    Color.__init__(self, 255, 255, 255)

class Texture:
    # Base class is null texture that reflects no light
    def getcolor(self, E, p, pobj, world, lights):
      return Color(0,0,0)
    
class Diffuse(Texture):
    # Simple ambient/diffuse texturing
    def __init__(self, color, ambient):
        self.color = color
        self.ambient = ambient
        self.diffuse = 1-ambient

    def getcolor(self, E, p, pobj, world, lights):
        res = self.color * self.ambient
        for light in lights:
            o = world.intersects(p, (light.loc-p).norm(), [pobj])
            if o is False or p.dist(light.loc) < p.dist(o[0]):
                res += light.color*light.intensity*pobj.shade(p, (light.loc-p).norm())*self.diffuse/(p.dist(light.loc)**2)
            elif isinstance(o[2].texture, Transparent): #TODO: check for partial transparency
                # Shadowed by transparent object
                #TODO: make separate Caustic texture
                #TODO: caustics outside shadows
                # Start by getting photons for the given light
                photons = light.getphotons(world, o[2])
                # Keep only those falling on this object
                photons = filter(lambda (php, phtext, phpobj): pobj is phpobj, photons)
                for ph in photons:
                    dist = abs(p - ph[0])
                    # Gaussian of distance is weight
                    w = e ** (-1*(float(dist) ** 2)) #TODO: gaussian coefficients
                    #TODO: consider angle of incidence
                    #TODO: transparency effect of refracting object
                    res += light.color*light.intensity*self.diffuse*w
        return res

class Specular(Texture):
    def __init__(self, specular):
        self.specular = specular

    def getcolor(self, E, p, pobj, world, lights):
        res = Color(0,0,0)
        for light in lights:
            o = world.intersects(p, (light.loc-p).norm(), [pobj])
            if o is False or p.dist(light.loc) < p.dist(o[0]):
                r = (light.loc-p).norm()
                N = pobj.normal(p)
                res += light.color*light.intensity*(pobj.shade(p, (N.scale((r*N)*2)-r).norm())**self.specular)/(p.dist(light.loc)**2)
        return res

class Reflective(Texture):
    def __init__(self):
        pass

    def getcolor(self, E, p, pobj, world, lights):
        r = (p-E).norm()
        N = pobj.normal(p)
        V = r-N.scale((r*N)*2)
        res = Ray(p+V.scale(EPSILON), V).render(world, lights)
        return res

class Transparent(Texture):
    def __init__(self, color, alpha, ir):
        self.color = color
        self.alpha = alpha
        self.ir = ir

    def getcolor(self, E, p, pobj, world, lights):
        try:
            p, V, l = pobj.get_refracted_ray(E, p, self.ir)
        except:
            return Color(0,0,0)
        res = Ray(p, V).render(world, lights, [pobj])
        if res is False:
            return SkyColor()
        if l == 0:
            p = 0
        else: 
            p = (1-self.alpha)**(1/l)
        return res*(1-p) + self.color*p #TODO: fix this awful transparency model

class Checkers(Texture):
    def __init__(self, tex1, tex2, size=[1,1]):
        self.tex1 = tex1
        self.tex2 = tex2
        
        if not isinstance(size, list) or len(size) != 2: raise ValueError
        self.u, self.v = size
        self.u = 1/float(self.u)
        self.v = 1/float(self.v)

    def getcolor(self, E, p, pobj, world, lights):
        if int(floor(pobj.uv(p)[0].scale(self.u)*p) + floor(pobj.uv(p)[1].scale(self.v)*p)) % 2 == 0:
            return self.tex1.getcolor(E, p, pobj, world, lights)
        else:
            return self.tex2.getcolor(E, p, pobj, world, lights)

class CombTexture(Texture):
    def __init__(self, *t):
        self.textures = t

    def getcolor(self, E, p, pobj, world, lights):
        res = Color(0,0,0)
        for t,w in self.textures:
            r = t.getcolor(E, p, pobj, world, lights)
            if r is False:
                continue
            res += r*w
        return res

class Object:
    def __init__(self, loc):
        self.loc = loc

    def intersects(self, E, V):
        return False

    def is_interior(self, p):
        return False

    def shade(self, P, V):
        raise NotImplementedError

    def normal(self, P):
        # Returns surface normal vector at a given point
        raise NotImplementedError

    def uv(self, p):
        # Returns surface u and v vectors for a given point
        raise NotImplementedError

    def get_refracted_ray(self, E, p, ir):
        # Used for transparent materials
        raise NotImplementedError

class Union(Object):
    def __init__(self, *objs):
        self.objs = objs

    def intersects(self, E, V, ignore=[]):
        r = False
        for obj in self.objs:
            if obj in ignore:
                continue
            t = obj.intersects(E, V)
            if r is False:
                r = t
                continue
            if t is not False and E.dist(t[0]) < E.dist(r[0]):
                r = t
        return r

    def is_interior(self, P):
        for obj in self.objs:
            if obj.is_interior(P): return True
        return False

class Intersection(Object):
    def __init__(self, a, b, texture):
        self.a = a
        self.b = b
        self.texture = texture

        self.ps = {}

    def intersects(self, E, V, ignore=[]):
        if self in ignore:
            return False
        
        ia, ib = self.a.is_interior(E), self.b.is_interior(E)

        r = Union(self.a, self.b).intersects(E, V)
        if r is False:
            return False
        p, _, pobj = r

        if (ia and pobj is self.b) or (ib and pobj is self.a):
            self.ps[p] = pobj
            return (p, self.texture, self)
        else:
            return self.intersects(p+V.scale(EPSILON), V, ignore)

    def is_interior(self, P):
        return self.a.is_interior(P) and self.b.is_interior(P)

    def normal(self, p):
        return self.ps[p].normal(p)

    def shade(self, P, V):
        return self.ps[P].shade(P, V)

class Sphere(Object):
    def __init__(self, loc, radius, texture):
        Object.__init__(self, loc)
        self.r = radius
        self.texture = texture
        
    def intersects(self, E, V, ignore=[]):
        if self in ignore:
            return False
        if self.is_interior(E):
            r = self.inner_intersects(E, V)
            if r is False:
                return False
            return (r, self.texture, self)
        EO = self.loc - E
        v = EO * V.norm()
        d = (self.r**2) - (abs(EO)**2 - v**2)
        if d<0:
            return False
        else:
            s = v-sqrt(d)
            if s <= 0:
                return False
            P = E + V.norm().scale(s)
            return (P, self.texture, self)

    def inner_intersects(self, E, V):
        EO = self.loc - E
        v = EO * V.norm()
        d = (self.r**2) - (abs(EO)**2 - v**2)
        if d<=0:
            return False
        else:
            s = v+sqrt(d)
            P = E + V.norm().scale(s)
            return P

    def is_interior(self, P):
        return self.loc.dist(P) < self.r

    def normal(self, P):
        return (P-self.loc).norm()

    def shade(self, P, V):
        return max([V.norm()*(P-self.loc).norm(),0])

    def get_refracted_ray(self, E, p, ir):
        V = (p-E).norm()

        if self.is_interior(E):
            rV = V
            np = p
        else:
            N = self.normal(p)
            c1 = -(V*N)
            c2 = sqrt(1 - (1/(ir**2))*(1-c1**2))

            if c1 >= 0:
                rV = V.scale(1/ir) + N.scale((c1 / ir) - c2 )
            else:
                rV = V.scale(1/ir) - N.scale((c1 / ir) - c2 )

            np = self.inner_intersects(p, rV)

        N = self.normal(np)
        c1 = -(rV*N)
        try:
            c2 = sqrt(1 - (ir**2)*(1-c1**2))

            if c1 >= 0:
                rV = V.scale(ir) + N.scale((c1 * ir) - c2 )
            else:
                rV = V.scale(ir) - N.scale((c1 * ir) - c2 )
        except ValueError:
            raise NotImplementedError # Totally internal reflection
        return np, rV, p.dist(np)
        
class Plane(Object):
    def __init__(self, loc, normal, texture):
        Object.__init__(self, loc)
        self.norm = normal.norm()
        self.texture = texture

        self.u = Vector(1,0,0)
        self.u = (self.u - self.norm.scale(self.u*self.norm)).norm()
        self.v = self.norm.cross(self.u).norm()

    def intersects(self, E, V, ignore=[]):
        if self in ignore or V*self.norm == 0 or (E-self.loc)*self.norm == 0:
            return False
        d = (self.loc-E)*self.norm / (V*self.norm)
        if d <= 0:
            return False
        P = E + V.norm().scale(d)
        return (P, self.texture, self)

    def shade(self, P, V):
        return max([V.norm()*self.norm.norm(),0])

    def normal(self, P):
        return self.norm.norm()

    def get_refracted_ray(self, E, p, ir):
        V = (p-E).norm()
        return p + V.scale(EPSILON), V , EPSILON

    def uv(self, P):
        return self.u, self.v

class HalfSpace(Plane):
    def is_interior(self, P):
        return (self.norm * (P-self.loc)) < 0

class Inverse(Object):
    def __init__(self, obj, text):
        self.obj = obj
        self.texture = text

    def intersects(self, E, V, ignore=[]):
        if self in ignore:
            return False
        r = self.obj.intersects(E, V, ignore)
        if r is False:
            return False
        return (r[0], self.texture, self)

    def is_interior(self, P):
        return not self.obj.is_interior(P)

    def normal(self, P):
        return self.obj.normal(P).scale(-1)

    def shade(self, P, V):
        return self.obj.shade(P, V.scale(-1))
    
class Light(Object):
    nphotons = 8192

    def __init__(self, loc, color, intensity):
        Object.__init__(self, loc)
        self.color = color
        self.intensity = intensity
        self.photons = {}

    def getphotons(self, world, transp_obj):
        key = world, transp_obj
        if self.photons.get(key, None) is not None:
            return self.photons[key]
        photons = []
        self.photons[key] = photons
        for V in self.makephotons(self.nphotons):
            r = transp_obj.intersects(self.loc, V)
            if r is not False:
                p, text, pobj = r
                np, nV, l = transp_obj.get_refracted_ray(self.loc, p, transp_obj.texture.ir) #TODO: handle partial transparency -- this assumes isinstance(transp_obj.texture, Transparent)
                nr = world.intersects(np, nV)
                if nr is not False:
                    photons.append(nr) #TODO: need to indicate length of internal ray
        return photons

    def makephotons(self, n):
        """
        n points distributed evenly on the surface of a unit sphere
        from http://stackoverflow.com/questions/14805583/dispersing-n-points-uniformly-on-a-sphere?lq=1
        """ 
        z = 2 * numpy.random.rand(n) - 1   # uniform in -1, 1
        t = 2 * pi * numpy.random.rand(n)   # uniform in 0, 2*pi
        x = sqrt(1 - z**2) * cos(t)
        y = sqrt(1 - z**2) * sin(t)

        result = []
        for i in range(n):
            result.append(Vector(x[i], y[i], z[i]))

        return result

class Camera(Object):
    def __init__(self, loc, aim, u=160):
        Object.__init__(self, loc)
        self.centerline = aim.norm()

        self.width = u
        self.height = u/16*9

    def render(self, obj, lights):
      try:
        im = Image.new("RGB", (self.width, self.height), tuple(SkyColor()))
        for i in range(self.height):
            for j in range(self.width):
                d = 1.0/self.width
                V = Vector((j - (self.width-1)/2)*d, (-i + (self.height-1)/2)*d, -1 ).norm()
                r = Ray(self.loc, self.centerline.rotate(V)).render(obj, lights)
                im.putpixel((j,i), tuple(r))
      finally:
        im.save("out.png", "png")

class Ray:
    def __init__(self, E, V):
        self.E = E
        self.V = V

    def render(self, obj, lights, ignore=[]):
        r = obj.intersects(self.E, self.V, ignore)
        if r is False:
            return SkyColor()
        p, text, pobj = r
        if p == self.E:
            return Color(0,0,0)
        return text.getcolor(self.E, p, pobj, obj, lights)
        
if __name__ == "__main__":
    c = Camera(Coord(0,2.5,0), Vector(0,-.5,-1), 64*3)
    c.render(
      Union(
        Plane(Coord(0,-2,0), Vector(0,1,0),
          Checkers(Diffuse(Color(255,255,255), .2), Diffuse(Color(0,0,0), .2))
        ),
#        Intersection(
#          Inverse(
#            Sphere(Coord(0,0,-5), .95, Texture()),
#            Texture()
#          ),
#          Intersection(
#            Sphere(Coord(0,0,-5), 1, Texture()),
#            HalfSpace(Coord(0,0,0), Vector(0,1,0), Texture()),
#            Texture()
#          ),
#          CombTexture(
#            (Reflective(), .2),
#            (Diffuse(Color(255,0,0), .2), .8),
#            (Specular(30), .8),
#          )
#        ),
        Sphere(Coord(-.2,.5,-5), .33,
          Transparent(Color(0,255,0), .8, 1.1),
#          CombTexture(
#            (Transparent(Color(0,255,0), .8, 1.1), .7),
#            (Reflective(), .1),
#            (Diffuse(Color(0,255,0), .2), .2),
#            (Specular(35), .4),
#          )
        ),
      ),
      [
        Light(Coord(-1,1.5,-2), Color(32,255,32), 5),
        Light(Coord(0,4,-3), Color(32,32,255), 20),
      ])
