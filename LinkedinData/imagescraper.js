let imglist = document.querySelector('.social-details-reactors-tab-body__actor-list').querySelectorAll('img')
let urls = []
for (let i = 0; i < imglist.length; i++) {
  urls.push(imglist[i].src)
}