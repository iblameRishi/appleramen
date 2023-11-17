import reflex as rx
import os
import shutil
from appleramen import model
from appleramen import appleramen

UPLOAD_DIR = './upload/class/'


class State(rx.State):
    # Store image file uploaded in local directory /upload/class/
    pred: str

    async def handle_upload(self, files: list[rx.UploadFile]):
        filename = ''
        for file in files:
            filename = file.filename
            upload_data = await file.read()
            outfile = f".web/public/{filename}"

            with open(outfile, "wb") as file_object:
                file_object.write(upload_data)

        for file in os.listdir('.web/public/'):
            if file.endswith('.jpg') or file.endswith('jpeg') or file.endswith('.png'):
                shutil.copy(outfile, UPLOAD_DIR)

        model1, model2, model3 = model.create_models()
        path = './upload'
        self.pred = str(model.predict(path, model1, model2, model3))
        os.remove(outfile)
        os.remove(f'./upload/class/{filename}')
