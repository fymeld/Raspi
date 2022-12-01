import numpy
from numpy import dot
from numpy.linalg import norm
def face_distance(faces_to_copare,face):
    cos_dist=[]
    y=face
    for x in faces_to_copare:
        cos_dist.append(numpy.inner(x,y)/(norm(x)*norm(y)))

    return [i.item() for i in cos_dist] 