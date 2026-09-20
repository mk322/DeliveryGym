(()=>{
  const $=selector=>document.querySelector(selector);
  const dialog=$('#image-dialog');
  let opener;
  document.addEventListener('click',event=>{
    const button=event.target.closest('[data-zoom]');
    if(!button)return;
    opener=button;
    const source=button.dataset.zoom;
    $('#dialog-image').src=source;
    $('#dialog-image').alt=button.querySelector('img')?.alt||'Paper figure';
    $('#full-image').href=source;
    dialog.showModal();
    $('#close-dialog').focus();
  });
  $('#close-dialog').addEventListener('click',()=>dialog.close());
  dialog.addEventListener('click',event=>{
    if(event.target!==dialog)return;
    const box=dialog.getBoundingClientRect();
    if(event.clientX<box.left||event.clientX>box.right||event.clientY<box.top||event.clientY>box.bottom)dialog.close();
  });
  dialog.addEventListener('close',()=>opener?.focus());

  const bibtex=`@misc{kang2026deliverygym,
  title={DeliveryGym: An RL Environment for Long-Horizon Embodied Agent Planning with Adaptive Curriculum},
  author={Haoqiang Kang and Yiming Zhang and Yiyang Guo and Chuying Li and Jianzhi Shen and Tianruo Rose Xu and Xiaokang Ye and Lianhui Qin},
  year={2026},
  howpublished={Project website},
  url={https://mk322.github.io/DeliveryGym/}
}`;
  $('#bibtex').textContent=bibtex;
  $('#copy-citation').addEventListener('click',async()=>{
    try{
      await navigator.clipboard.writeText(bibtex);
      $('#copy-feedback').textContent='BibTeX copied to clipboard.';
      $('#copy-citation').textContent='Copied ✓';
    }catch{
      const range=document.createRange();
      range.selectNodeContents($('#bibtex'));
      const selection=window.getSelection();
      selection.removeAllRanges();selection.addRange(range);
      $('#copy-feedback').textContent='Citation selected. Press Ctrl+C or ⌘C to copy.';
      $('#copy-citation').textContent='Select & copy';
    }
  });
})();
