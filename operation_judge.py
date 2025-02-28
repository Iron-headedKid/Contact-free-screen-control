import pyautogui

pyautogui.PAUSE = 0.1      # 点击停顿时间，单位s
pyautogui.FAILSAFE = True        # 自动防故障功能（必须）


# 点击判定
# (x0,y0)为前一时刻坐标，(x1,y1)为当前时刻坐标
# 点击判定逻辑：当手指在探测范围消失即判定为点击，点击位置为(x0,y0)
# 当手指消失时(x1,y1)为(0,0)
def judge(x1, y1, x0, y0):
    if x0 != 0 and y0 != 0:
        if x0 - x1 == x0 and y0 - y1 == y0:
            if y0 < 150:
                x = int(1.9*x0 + 1.707*y0 - (x0*y0)/325 - 256)
                y = int(1.5*y0)
            if y0 >= 150:
                x, y = int(1.5*x0), int(1.5*y0)
            pyautogui.click(x=x, y=y)
            pyautogui.click(x=x, y=y)
            print('click:', x, y)
    x0, y0 = x1, y1
    return x0, y0
