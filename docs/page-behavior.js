(function(){'use strict';
 const toggle=document.getElementById('menuButton'); const nav=document.getElementById('nav');
 function setNav(open){if(!nav||!toggle)return;nav.classList.toggle('is-open',open);toggle.setAttribute('aria-expanded',String(open));toggle.setAttribute('aria-label',open?'Close navigation menu':'Open navigation menu');toggle.textContent=open?'×':'☰';}
 if(toggle)toggle.addEventListener('click',function(){setNav(!nav.classList.contains('is-open'));});
 if(nav)nav.querySelectorAll('a').forEach(a=>a.addEventListener('click',()=>setNav(false)));
 document.addEventListener('keydown',e=>{if(e.key==='Escape')setNav(false);});
 document.querySelectorAll('.copy').forEach(b=>b.addEventListener('click',async function(){
 const code=this.closest('.code-panel').querySelector('code');const original=this.getAttribute('data-original')||this.textContent;this.setAttribute('data-original',original);
 try{if(!navigator.clipboard||!window.isSecureContext)throw Error('Clipboard unavailable');await navigator.clipboard.writeText(code.innerText);this.textContent='Copied';}
 catch(e){const r=document.createRange(),s=window.getSelection();r.selectNodeContents(code);s.removeAllRanges();s.addRange(r);this.textContent='Selected';}
 this.setAttribute('aria-label',this.textContent+' commands');
 }));
})();
