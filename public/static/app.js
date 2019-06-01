const app = new Vue({
    el: '#app',
    data: {
        // 所有圖片
        imageList: [],

        // 原圖
        originalImage: '',
        // 處理過的圖片
        processedImage: '',

        // 除錯用的圖片
        devImages: [],

        // 圖片處理中
        processing: true,
        is_dragover: false,
        is_uploading: false,
        is_success: false,
        is_error: false,

        // 計時器
        timeout: 2000,
        uploadTimeout: null,
    },

    methods: {
        // 顯示所有圖片
        showImageList: function () {
            $.post('/', data => {
                this.imageList = JSON.parse(data)['images'];
            });
        },

        // 顯示處理過的圖片
        showProcessedImage: function (event) {
            this.processing = true;
            $('#ImageModalCarousel').carousel(1);

            var path = $(event.relatedTarget).data('src');
            this.originalImage = this.processedImage = path;

            $.post('/processed', { 'image': path }, data => {
                data = JSON.parse(data)

                this.devImages = data['dev_images'] || [];
                this.processedImage = data['image'];
                this.processing = false;
            });
        },

        // 拖拉上傳圖片用的計時器
        uploadTimeoutReset: function (onlyClear = false) {
            clearTimeout(this.uploadTimeout);

            if (!onlyClear) {
                this.uploadTimeout = setTimeout(() => {
                    this.is_dragover = false;
                    $('#UploadModal').modal('hide');
                }, this.timeout);
            }
        },

        // 拖拉上傳圖片顯示Modal
        appDragover: function () {
            $('#UploadModal').modal('show');

            this.uploadTimeoutReset();
        },

        // 拖拉到可上傳的範圍
        uploadDragover: function () {
            this.is_dragover = true;
        },
        uploadDragleave: function () {
            this.is_dragover = false;
        },

        // 拖拉上傳圖片
        uploadDrop: function (event) {
            let images = event.dataTransfer.files;

            this.uploadInput(images);
        },

        // 點擊上傳圖片
        uploadInput: function (images = null) {
            const image_type = ['image/jpeg', 'image/png', 'image/gif', 'image/bmp', 'image/tiff'];
            images = images || $('#image')[0].files;

            if (images.length > 0 && image_type.includes(images[0].type)) {
                this.uploadTimeoutReset(true);
                this.is_uploading = true;

                let form = new FormData();
                form.append('image', images[0]);
    
                $.ajax({
                    url: '/upload',
                    type: 'POST',
                    data: form,
                    contentType: false,
                    processData: false,
                    success: data => {
                        this.imageList.push(JSON.parse(data))

                        this.is_uploading = false;
                        this.is_success = true;
                    },
                });
            } else {
                this.is_error = true;
            }
        },

        uploadStatusReset: function () {
            this.is_uploading = false;
            this.is_success = false;
            this.is_error = false;
            
            $('#image')[0].value = '';
        },
    },

    mounted: function () {
        $('#ImageModal').on('show.bs.modal', this.showProcessedImage);
        $('#ImageModal').on('hidden.bs.modal', () => { this.devImages = [] });
        $('#UploadModal').on('hidden.bs.modal', this.uploadStatusReset);

        this.showImageList();
    },
});
