from django.shortcuts import render
import requests
from subprocess import run, PIPE
import sys
import re
import pandas as pd

df = pd.read_csv('/temporaryProject/kota kec.csv')
kota = sorted(set(df.Kota.values))
def home(request):
    return render(request, 'mapping.html', {'kota_latlong':kota})

def external(request):
    inp1 = request.POST.get('bencana')
    inp2 = request.POST.get('since')
    inp3 = request.POST.get('until')
    inp4 = request.POST.get('kecamatan')
    inp5 = request.POST.get('kota')
    out = run([sys.executable, '/temporaryProject/learning.py', inp1, inp2, inp3, inp4, inp5], shell=False, stdout=PIPE, encoding='utf-8')
    output = out.stdout

    fin1 = re.findall(r'idx:.+', output)

    user_twt = []
    koordinat = []

    for i in fin1:

        a = re.findall(r'idx:(\d+) usr:(.+) tweet:(.+) \[([\-.0-9].+), ([0-9.].+)\]', i)
        user_twt.append([a[0][0], a[0][1], a[0][2]])
        koordinat.append([float(a[0][3]), float(a[0][4])])
    return render(request, 'mapping.html', {'user_twt': user_twt,
                                            'koordinat': koordinat,
                                            'kota_latlong': kota})
