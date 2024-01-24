function demo_load(x) {
    document.body.scrollTop = document.documentElement.scrollTop = 0;

    function gradioApp() {
        const elems = document.getElementsByTagName('gradio-app');
        const elem = elems.length == 0 ? document : elems[0];
    
        if (elem !== document) {
            elem.getElementById = function(id) {
                return document.getElementById(id);
            };
        }
        return elem.shadowRoot ? elem.shadowRoot : elem;
    }

    function all_gallery_buttons() {
        var allGalleryButtons = gradioApp().querySelectorAll('#outputgallery .thumbnail-item.thumbnail-small');
        var visibleGalleryButtons = [];
        allGalleryButtons.forEach(function(elem) {
            if (elem.parentElement.offsetParent) {
                visibleGalleryButtons.push(elem);
            }
        });
        return visibleGalleryButtons;
    }
    
    function selected_gallery_button() {
        return all_gallery_buttons().find(elem => elem.classList.contains('selected')) ?? null;
    }
    
    function selected_gallery_index() {
        return all_gallery_buttons().findIndex(elem => elem.classList.contains('selected'));
    }

    function loadImg(src){
        return new Promise((resolve, reject) => {
          let img = new Image()
          img.onload = () => resolve(img)
          img.onerror = reject
          img.src = src
        })
    }

    async function resize_b64_img(b64_img, max_side=2048) {
        var img = await loadImg(b64_img);
        naturalWidth = img.naturalWidth;
        naturalHeight = img.naturalHeight;

        if (naturalWidth > max_side || naturalHeight > max_side) {
            var width = 0;
            var height = 0;
            if (naturalWidth >= naturalHeight) {
                width = max_side;
                height = Math.ceil((max_side / naturalWidth) * naturalHeight);
            } else {
                height = max_side;
                width = Math.ceil((max_side / naturalHeight) * naturalWidth);
            }

            var canvas = document.createElement('canvas');
            ctx = canvas.getContext('2d');
            canvas.width = width;
            canvas.height = height;
            ctx.drawImage(img, 0, 0, width, height);
            return canvas.toDataURL();
        }
        return b64_img;
    }

    // fix image preview on mobile
    function imageMaskResize() {
        const canvases = gradioApp().querySelectorAll('#inputmask canvas');
        if (!canvases.length) {
            window.removeEventListener('resize', imageMaskResize);
            return;
        }
    
        const wrapper = canvases[0].closest('.wrap');
        const previewImage = wrapper.previousElementSibling;
    
        if (!previewImage.complete) {
            previewImage.addEventListener('load', imageMaskResize);
            return;
        }
    
        const w = previewImage.width;
        const h = previewImage.height;
        const nw = previewImage.naturalWidth;
        const nh = previewImage.naturalHeight;
        const portrait = nh > nw;
    
        const wW = Math.min(w, portrait ? h / nh * nw : w / nw * nw);
        const wH = Math.min(h, portrait ? h / nh * nh : w / nw * nh);
    
        wrapper.style.width = `${wW}px`;
        wrapper.style.height = `${wH}px`;
        wrapper.style.left = `0px`;
        wrapper.style.top = `0px`;
    
        canvases.forEach(c => {
            c.style.width = c.style.height = '';
            c.style.maxWidth = '100%';
            c.style.maxHeight = '100%';
            c.style.objectFit = 'contain';
        });
    }

    window.gradioApp = gradioApp
    window.all_gallery_buttons = all_gallery_buttons
    window.selected_gallery_button = selected_gallery_button
    window.selected_gallery_index = selected_gallery_index
    window.resize_b64_img = resize_b64_img
    window.imageMaskResize = imageMaskResize;

    window.addEventListener('resize', imageMaskResize);
}