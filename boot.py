import pymongo
from pymongo import MongoClient
import base64
import os


class Boot(object):

    def __init__(self, db, training_folder):
        self.employee = 0
        self.blacklist = 0
        self.visitor_faces = []
        self.training_folder = training_folder
        self.db = db
        self.index = 1
        self.labels = [""]
        self.init_training_data(training_folder)

    def init_training_data(self, top):
        """
        TODO: comment description ??????


        :param top:
        :return:
        """
        for root, dirs, files in os.walk(top, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))

    def _generate_imgs(self, collection):
        """
        TODO: comment description ??????



        :param main_path:
        :param database:
        :param collection:
        :return:
        """
        client = MongoClient('localhost', 27017)
        db = client[self.db]
        col = db[collection]
        recs = col.find()
        len = recs.count()
        for rec in recs:
            ind = 1
            self.labels.append(rec['name'])
            file_path = os.path.join(self.training_folder, 'p' + str(self.index))
            os.mkdir(file_path)
            img = base64.b64decode(rec['image'])
            with open(os.path.join(file_path, '{}.jpg'.format(ind)), 'wb') as f:
                f.write(img)
                ind += 1
            self.index += 1
        return len

    def generate_training_data(self):
        return self._generate_imgs('employees'),  self._generate_imgs('blaclist')


if __name__ == '__main__':
    boot = Boot('jaw', 'training-data')
    nbr = boot.generate_training_data()
