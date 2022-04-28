from config_data import cari_kec, cari_kot
from bersihin import bersihin
import pandas as pd
import random

kec = pd.read_csv('/temporaryProject/kota kec.csv')
kot = pd.read_csv('/temporaryProject/cities.csv')
def search_kec(keyword, since, until, nama_wilayah):
    data = []

    if nama_wilayah.upper() in kec.Kecamatan.values:
        for i in cari_kec(keyword, since, until, str(kec.loc[kec['Kecamatan'] == nama_wilayah.upper()].Latitude.values[0]), str(kec.loc[kec['Kecamatan'] == nama_wilayah.upper()].Longitude.values[0])):
            try:
                if len(bersihin(i.tweet)) < 10 or bersihin(i.tweet) == '':
                    continue
                data.append([i.username, bersihin(i.tweet), [i.place['coordinates'][0], i.place['coordinates'][1]]])
            except:
                if len(bersihin(i.tweet)) < 10 or bersihin(i.tweet) == '':
                    continue
                data.append([i.username, bersihin(i.tweet), [kec.loc[kec['Kecamatan'] == nama_wilayah.upper()].Latitude.values[0]+random.uniform(-0.01,0.01), kec.loc[kec['Kecamatan'] == nama_wilayah.upper()].Longitude.values[0]+random.uniform(-0.01,0.01)]])

        df = pd.DataFrame(data, columns=['username','tweet','addressPoint'])
        return df
    else:
        print('nama wilayah tidak ditemukan')
def search_kot(keyword, since, until, nama_wilayah):
    data = []

    if nama_wilayah.upper() in kot.Name.values:
        for i in cari_kot(keyword, since, until, str(kot.loc[kot['Name'] == nama_wilayah.upper()].Latitude.values[0]), str(kot.loc[kot['Name'] == nama_wilayah.upper()].Longitude.values[0])):
            try:
                if len(bersihin(i.tweet)) < 10 or bersihin(i.tweet) == '':
                    continue
                data.append([i.username, bersihin(i.tweet), [i.place['coordinates'][0], i.place['coordinates'][1]]])
            except:
                if len(bersihin(i.tweet)) < 10 or bersihin(i.tweet) == '':
                    continue
                data.append([i.username, bersihin(i.tweet), [kot.loc[kot['Name'] == nama_wilayah.upper()].Latitude.values[0]+random.uniform(-0.02,0.02), kot.loc[kot['Name'] == nama_wilayah.upper()].Longitude.values[0]+random.uniform(-0.04,0.02)]])

        df = pd.DataFrame(data, columns=['username','tweet','addressPoint'])
        return df
    else:
        print('nama wilayah tidak ditemukan')
