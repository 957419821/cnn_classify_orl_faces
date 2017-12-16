import numpy as np

def initialImage():
    def blanks():
        def blank():
            return [0. for i in range(10)]
        return [blank(), blank()]
    def sandwiches():
        def getSandwich():
            return [0., 0.] + [100. for i in range(6)] + [0., 0.]
        return [getSandwich() for i in range(6)]
    return np.mat(blanks() + sandwiches() + blanks()), np.rint((np.matlib.rand(2) - 0.5) * 10)

def translation(image, vector):
    vertical, horizon = np.array(vector)[0]; vertical = int(vertical); horizon = int(horizon)
    rowLength = len(image); columnLength = len(np.array(image[0])[0])
    columnPadding = np.rint(np.matlib.randn((rowLength, np.abs(horizon))) * 4 + 50, casting = 'int', dtype=np.int8)
    rowPadding = np.rint(np.matlib.randn((np.abs(vertical), columnLength)) * 4 + 50, casting = 'int', dtype=np.int8)
    translatedImage = image
    if horizon > 0:
        translatedImage = np.column_stack((columnPadding, translatedImage[:, 0:-horizon]))
    elif horizon < 0:
        translatedImage = np.column_stack((translatedImage[:, -horizon:], columnPadding))
    if vertical > 0:
        translatedImage = np.row_stack((rowPadding, translatedImage[0:-vertical, :]))
    if vertical < 0:
        translatedImage = np.row_stack((translatedImage[-vertical:, :], rowPadding))
    return translatedImage
