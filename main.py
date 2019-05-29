from ImageProcessing import handle, get_test_image

from jinja2 import Environment, FileSystemLoader

from tornado import ioloop, web

# 設定網頁伺服器的埠口
PORT = 8000

# Jinja載入index.html
templateEnv = Environment(loader=FileSystemLoader("public/"))
template = templateEnv.get_template("index.html")


def main():
    html = template.render(images=handle(get_test_image('19')))

    # 頁面處理
    class Handler(web.RequestHandler):
        def get(self):
            self.write(html)

    # 分配Handler
    web.Application([
        (r"/", Handler),
    ]).listen(PORT)

    print('Server start on:', 'localhost:' + str(PORT))
    ioloop.IOLoop.instance().start()


if __name__ == '__main__':
    main()
