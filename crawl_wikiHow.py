#!/usr/bin/python

import sys
import time
import urllib
import codecs

class AppURLopener(urllib.FancyURLopener):
    version = "Mozilla/5.0 (Windows; U; Windows NT 5.1; it; rv:1.8.1.11) Gecko/20071127 Firefox/2.0.0.11"
    
def crawl_from_base_cat(cat_buffer):
    crawled_cat = set()
    crawled_pages = set()
    urllib._urlopener = AppURLopener()
    query = "https://www.wikihow.com/index.php?title=Special:Export"
    while cat_buffer:
        category = cat_buffer.pop(0)
        data = {'catname':category, 'addcat':'', 'wpDownload':1, 'action':'submit'}
        data = urllib.urlencode(data)
        page = urllib.urlopen(query, data)
        begin_cat = False
        keep_looking = True
        pages_to_crawl = []
        for line in page:
            if not keep_looking:
                time.sleep(3)
                break
            if begin_cat:
                if '</label>' in line:
                    begin_cat = False
                    keep_looking = False
                    new_page = line.strip().split('<', 1)[0]
                else:
                    new_page = line.strip()
                if new_page.startswith('Category:'):
                    new_page = new_page.split(':', 1)[1]
                    if new_page not in crawled_cat:
                        cat_buffer.append(new_page)
                        crawled_cat.add(new_page)
                else:
                    if new_page not in crawled_pages:
                        pages_to_crawl.append(new_page)
                        crawled_pages.add(new_page)
            elif '<label for="catname">' in line:
                begin_cat = True
        if not pages_to_crawl:
            continue
        with open('%s.xml' %category, 'w') as fp:
            sys.stderr.write('Category :: %s\n' %category)
            sys.stderr.write('\n'.join(['    %s' %pg for pg in pages_to_crawl]))
            sys.stderr.write('\n\n%s\n\n' %('='*70))
            pages_to_crawl = '\n'.join(pages_to_crawl)
            data = {'wpDownload':1, 'action':'submit', 'pages':pages_to_crawl}
            data = urllib.urlencode(data)
            page = urllib.urlopen(query, data)
            fp.write(page.read())
        time.sleep(3)

def main():
    base_cat = '''
        Arts and Entertainment
        Cars & Other Vehicles
        Computers and Electronics
        Education and Communications
        Family Life
        Finance and Business
        Food and Entertaining
        Health
        Hobbies and Crafts
        Holidays and Traditions
        Home and Garden
        Personal Care and Style
        Pets and Animals
        Philosophy and Religion
        Relationships
        Sports and Fitness
        Travel
        Work World
        Youth
        '''
    base_cat = [x.strip() for x in base_cat.strip().split('\n')]
    crawl_from_base_cat(base_cat)

if __name__ == '__main__':
    main()
