"""
Week 4: DFO for finding the brightest area in an image

Based on the lecturer's DFO code (github.com/mohmaj/DFO, Python/DFO.py). Changes:
  - each fly is a 2D position (x, y) inside the image, so D = 2
  - fitness = brightness of the image around that position (3x3 average),
    which is MAXIMISED, so argmin becomes argmax and < becomes >
  - the image can change between iterations (image_at(itr)), so moving,
    fading or switching images can be tested
  - fly positions are recorded every iteration for the figures
"""
import numpy as np
from scipy.ndimage import uniform_filter


# images
def blob(H, W, cx, cy, sigma, amp):
    yy, xx = np.mgrid[0:H, 0:W]
    return amp * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))


def synthetic_image(seed, H=200, W=200, n_dim=12, bright=True):
    """Cloudy grey background of dim blobs, plus one clearly brightest blob."""
    rng = np.random.default_rng(seed)
    img = np.zeros((H, W))
    for _ in range(n_dim):
        img += blob(H, W, rng.uniform(0, W), rng.uniform(0, H),
                    rng.uniform(12, 35), rng.uniform(0.15, 0.45))
    peak = None
    if bright:
        peak = (rng.uniform(25, W - 25), rng.uniform(25, H - 25))
        img += blob(H, W, *peak, 12, 1.0)
    img = 255 * img / img.max()
    return img, peak


def real_image(name, size=200):
    """Greyscale sample image from scikit-image, resized to about `size` pixels high."""
    from skimage import data, color, transform
    im = getattr(data, name)()
    if im.ndim == 3:
        im = color.rgb2gray(im)
    im = transform.resize(im, (size, int(size * im.shape[1] / im.shape[0])), anti_aliasing=True)
    return 255 * (im - im.min()) / (im.max() - im.min())


def brightness_map(img):
    """Average brightness in a 3x3 window, so single noisy pixels do not count as 'areas'."""
    return uniform_filter(img, size=3, mode='nearest')


# DFO
def dfo_image(image_at, N=30, maxIterations=100, delta=0.001, seed=0):
    """
    image_at(itr) returns the image to search at iteration itr.
    Returns a dict with the best position and brightness per iteration and all fly positions.
    """
    rng = np.random.default_rng(seed)
    D = 2
    H, W = image_at(0).shape
    lowerB = [0, 0]
    upperB = [W - 1, H - 1]

    def f(B, x):           # FITNESS = BRIGHTNESS AT (x, y)
        return B[int(round(x[1])), int(round(x[0]))]

    X = np.empty([N, D])
    fitness = [None] * N
    for i in range(N):
        for d in range(D):
            X[i, d] = rng.uniform(lowerB[d], upperB[d])

    best_pos, best_fit, positions = [], [], []
    for itr in range(maxIterations):
        B = brightness_map(image_at(itr))
        for i in range(N):                   # EVALUATION
            fitness[i] = f(B, X[i, ])
        s = int(np.argmax(fitness))          # FIND BEST FLY (BRIGHTEST)
        best_pos.append(X[s].copy()); best_fit.append(fitness[s]); positions.append(X.copy())

        for i in range(N):
            if i == s:
                continue       # ELITIST STRATEGY
            left, right = (i - 1) % N, (i + 1) % N
            bNeighbour = right if fitness[right] > fitness[left] else left
            for d in range(D):
                if rng.random() < delta:       # DISTURBANCE
                    X[i, d] = rng.uniform(lowerB[d], upperB[d])
                    continue
                u = rng.random()
                X[i, d] = X[bNeighbour, d] + u * (X[s, d] - X[i, d])
                if X[i, d] < lowerB[d] or X[i, d] > upperB[d]:      # OUT OF BOUND CONTROL
                    X[i, d] = rng.uniform(lowerB[d], upperB[d])

    return {'best_pos': np.array(best_pos), 'best_fit': np.array(best_fit),
            'positions': np.array(positions)}


if __name__ == '__main__':
    img, peak = synthetic_image(seed=1)
    res = dfo_image(lambda itr: img, seed=0)
    B = brightness_map(img)
    ty, tx = np.unravel_index(np.argmax(B), B.shape)
    print('Brightest area (true):   x = %d, y = %d, brightness %.1f' % (tx, ty, B.max()))
    bx, by = res['best_pos'][-1]
    print('Brightest area (DFO):    x = %.1f, y = %.1f, brightness %.1f' % (bx, by, res['best_fit'][-1]))