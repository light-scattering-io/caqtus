from contextlib import closing
import numpy as np
import matplotlib.pyplot as plt

from pixelfly import PixelflyBoard, Mode, BinMode, PixelDepth

if __name__ == "__main__":
    with closing(PixelflyBoard(board_number=0)) as board:
        board.set_mode(mode=Mode.SW_TRIGGER | Mode.ASYNC_SHUTTER, exp_time=5, hbin=BinMode.BIN_1X,
                       vbin=BinMode.BIN_1X, gain=False, bit_pix=PixelDepth.BITS_12)
        board.start_camera()
        im = board.read_image(1000)
        board.stop_camera()

    plt.figure()
    plt.imshow(im.T[710:720, 290:300], origin="lower")
    print(np.sum(im.T[710:720, 290:300]))
    plt.colorbar()
    plt.show()
    # np.save('current_traps.npy', im)
