import imghdr, json
from uuid import uuid4
from os import path, listdir

from ImageProcessing import handle, im_show

from jinja2 import Environment, FileSystemLoader
from tornado import ioloop, web

# 根目錄
BASE_DIR = path.dirname(path.abspath(__file__))

# 圖片目錄
IMAGES_DIR = 'images/'

# 設定網頁伺服器的埠口
PORT = 8000

# Jinja載入index.html
templateEnv = Environment(loader=FileSystemLoader("public/"))
template = templateEnv.get_template("index.html")


# 頁面處理
class MainHandler(web.RequestHandler):
    def get(self):
        self.write(template.render())

    def post(self):
        images = [{
            'src': path.join(IMAGES_DIR, f),
            'filename': f
        } for f in listdir(IMAGES_DIR)
                  if imghdr.what(path.join(IMAGES_DIR, f))]

        self.write(json.dumps({'images': images}))


# 圖片輸出
class ProcessedHandler(web.RequestHandler):
    def post(self):
        image_path = get_image(self.get_argument('image'))
        image = handle(image_path)
        output = {'image': 'data:image/jpg;base64,' + image}

        dev_images = im_show()
        if dev_images:
            output['dev_images'] = [
                'data:image/jpg;base64,' + im for im in dev_images
            ]

        self.write(json.dumps(output))


# 圖片輸入
class UploadHandler(web.RequestHandler):
    def post(self):
        info = self.request.files['image'][0]
        filename = uuid4().hex + '.' + info['filename'].split('.')[-1]

        image_type = [
            'image/jpeg', 'image/png', 'image/gif', 'image/bmp', 'image/tiff'
        ]
        if any(info['content_type'] in t for t in image_type):
            file = path.join(IMAGES_DIR, filename)
            with open(file, 'wb') as f:
                f.write(info['body'])
            f.close()

            self.write(json.dumps({'src': file, 'filename': filename}))
        else:
            self.send_error(400)


# 取得圖片
def get_image(filename):
    return path.join(BASE_DIR, filename)


def main():
    # 分配Handler
    web.Application([
        (r'/static/(.*)', web.StaticFileHandler, {
            'path': path.join(BASE_DIR, 'public/static/')
        }),
        (r'/images/(.*)', web.StaticFileHandler, {
            'path': path.join(BASE_DIR, IMAGES_DIR)
        }),
        (r"/", MainHandler),
        (r"/processed", ProcessedHandler),
        (r"/upload", UploadHandler),
    ]).listen(PORT)

    print('Server start on:', 'localhost:' + str(PORT))
    ioloop.IOLoop.instance().start()


if __name__ == '__main__':
    main()
