// Modal loader for manuscript/abstract pages
(function(){
  function init(){
    const modal = document.getElementById('abstract-modal');
    if(!modal) return; // nothing to do
    const body = document.getElementById('abstract-modal-body');

    function openModal(){
      modal.setAttribute('aria-hidden','false');
      modal.style.display = 'block';
      const close = modal.querySelector('.modal-close');
      if(close) close.focus();
      document.addEventListener('keydown', onKeyDown);
    }
    function closeModal(){
      modal.setAttribute('aria-hidden','true');
      modal.style.display = 'none';
      body.innerHTML = '';
      document.removeEventListener('keydown', onKeyDown);
    }
    function onKeyDown(e){ if(e.key === 'Escape') closeModal(); }

    // close handlers
    modal.querySelectorAll('[data-close]').forEach(el=> el.addEventListener('click', closeModal));

    // event delegation: handle clicks on .post-tags anchors
    document.addEventListener('click', function(e){
      const a = e.target.closest && e.target.closest('.post-tags a.tag');
      if(!a) return;
      // only handle links to the conferences folder (match with or without leading slash)
      const href = a.getAttribute('href');
      if(!href || (!href.includes('/conferences/') && !href.includes('conferences/'))) return;

      // Only intercept HTML pages (so images, PDFs, etc. open in a new tab normally)
      const isHtml = !!href.match(/\.html?(?:$|[?#])/i);
      if(!isHtml) {
        // don't prevent default — let target="_blank" open the resource
        return;
      }

      e.preventDefault();
      body.textContent = 'Loading…';
      openModal();

      fetch(href).then(res=>{
        if(!res.ok) throw new Error('Network response was not ok');
        return res.text();
      }).then(html => {
        const parser = new DOMParser();
        const doc = parser.parseFromString(html, 'text/html');
        const main = doc.querySelector('main') || doc.querySelector('body');
        if(main) body.innerHTML = main.innerHTML;
        else body.innerHTML = html;
      }).catch(err=>{
        closeModal();
        window.open(href, '_blank', 'noopener');
      });
    });
  }

  if(document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
