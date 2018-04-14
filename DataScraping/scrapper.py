import bs4 as bs
import urllib.request

sauce = urllib.request.urlopen('http://www.casablanca-bourse.com/en/Negociation-History.aspx?Cat=24&IdLink=225').read()

soup = bs.BeautifulSoup(sauce, 'lxml')

print(soup.title.string)

print(soup.td)

