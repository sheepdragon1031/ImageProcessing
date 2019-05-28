from ImageProcessing import detect, profile, TEST_IMAGE, IMAGES

from jinja2 import Environment, FileSystemLoader

import tornado.ioloop
import tornado.web

# 設定網頁伺服器的埠口
PORT = 8000

# Jinja載入index.html
templateEnv = Environment(loader=FileSystemLoader("public/"))
template = templateEnv.get_template("index.html")


def main():
    detect(profile(TEST_IMAGE['2']))

    html = template.render(images=IMAGES)

    # 頁面處理
    class Handler(tornado.web.RequestHandler):
        def get(self):
            self.write(html)

    # 分配Handler
    tornado.web.Application([
        (r"/", Handler),
    ]).listen(PORT)

    tornado.ioloop.IOLoop.instance().start()


if __name__ == '__main__':
    main()
