import requests

def downloadimage(i, pic_url):
    with open(i+'.jpg', 'wb') as handle:
            response = requests.get(pic_url, stream=True)

            if not response.ok:
                print(response)

            for block in response.iter_content(1024):
                if not block:
                    break

                handle.write(block)
urls = []

def get_urls(folder):
    urllist = open(folder+'/urls.txt', 'r')
    urls = [line.split('\n,') for line in urllist.readlines()]
    print(urls[0][0])
    # for i, url in enumerate(urls):
    #     downloadimage(i, url)

get_urls('SpatialComputing')