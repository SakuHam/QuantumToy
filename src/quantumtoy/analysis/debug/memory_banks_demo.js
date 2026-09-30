// Classical memory banks. All copies share the same original detection.
Object.assign(TEXT.en, {
  memoryTitle:'Reference memories and repeated delayed reads',
  memoryIntro:'Copy each classical detector record into reference and ageing memories. Compare the last read with everything recovered so far. Each original event is counted once.',
  referenceCopiesLabel:'Reference memories', delayedCopiesLabel:'Delayed memories',
  memoryReadsLabel:'Reads per delayed memory', memorySpacingLabel:'Between reads / σT',
  memoryReadEfficiencyLabel:'Read success if still stored', referenceSurvivalLabel:'Reference reliability / copy',
  memoryLossLabel:'Loss in delayed memories', memoryIndependent:'Independent copy lifetimes',
  memoryShared:'Shared lifetime across copies', memoryPreparationsLabel:'Particle preparations',
  memoryCreatedLegend:'Grey: original records', memoryReferenceLegend:'Blue: reference bank',
  memoryLastLegend:'Pink: last delayed read', memoryLogLegend:'Amber: delayed read log',
  memoryUnionLegend:'Cyan: reference or read log',
  memoryCreatedLabel:'expected original records', memoryLastLabel:'expected at last read',
  memoryLoggedLabel:'expected in read log', memoryRecoveredLabel:'expected unique recovered',
  memoryFirst:'First read', memoryFinal:'Last read', memoryFromControl:'The readout-wait control above sets the first read.',
  memoryUnavailable:'Original records not recovered', memoryAbsent:'No original record',
  memoryCopiesRead:'Expected readable copies at last read (duplicates included)',
  memoryScope:'Expected counts, not a sampled run. Dark records are included. Reference copies have independent, delay-independent reliability. Delayed memories age from record creation; reads do not refresh or destroy them. Successful reads remain in an ideal external log. Read errors are independent. Shared loss affects the delayed bank only; it is not universal erasure of every trace. None of these controls changes the wavefunction or original detection law.',
  memoryTimeAxis:'record creation / T₀', memoryCountAxis:'expected records / bin',
});
Object.assign(TEXT.fi, {
  memoryTitle:'Vertailumuistit ja toistuva viivästetty luku',
  memoryIntro:'Kopioi jokainen klassinen detektoritietue vertailumuisteihin ja vanheneviin muisteihin. Vertaa viimeistä lukua kaikkeen siihen mennessä talteen saatuun. Kukin alkuperäinen tapahtuma lasketaan kerran.',
  referenceCopiesLabel:'Vertailumuistien määrä', delayedCopiesLabel:'Viivästettyjen muistien määrä',
  memoryReadsLabel:'Lukukertoja viivästettyä muistia kohden', memorySpacingLabel:'Lukujen väli / σT',
  memoryReadEfficiencyLabel:'Luku onnistuu, jos tieto säilyy', referenceSurvivalLabel:'Vertailukopion luotettavuus',
  memoryLossLabel:'Viivästettyjen muistien häviäminen', memoryIndependent:'Kopioiden itsenäiset eliniät',
  memoryShared:'Kopioiden yhteinen elinikä', memoryPreparationsLabel:'Hiukkasvalmistusten määrä',
  memoryCreatedLegend:'Harmaa: alkuperäiset tietueet', memoryReferenceLegend:'Sininen: vertailumuistit',
  memoryLastLegend:'Pinkki: viimeinen viivästetty luku', memoryLogLegend:'Keltainen: viivästettyjen lukujen loki',
  memoryUnionLegend:'Turkoosi: vertailumuistit tai lukuloki',
  memoryCreatedLabel:'alkuperäisiä tietueita, odotusarvo', memoryLastLabel:'viimeisessä luvussa, odotusarvo',
  memoryLoggedLabel:'lukulokissa, odotusarvo', memoryRecoveredLabel:'eri tietueita talteen, odotusarvo',
  memoryFirst:'Ensimmäinen luku', memoryFinal:'Viimeinen luku', memoryFromControl:'Ylempi lukuviivesäädin määrää ensimmäisen lukuhetken.',
  memoryUnavailable:'Alkuperäisiä tietueita saamatta talteen', memoryAbsent:'Ei alkuperäistä tietuetta',
  memoryCopiesRead:'Luettavia kopioita viimeisessä luvussa, odotusarvo (sisältää kaksoiskappaleet)',
  memoryScope:'Luvut ovat odotusarvoja, eivät arvottu koeajo. Pimeät tietueet ovat mukana. Vertailukopioiden luotettavuus on riippumaton muista kopioista ja viiveestä. Viivästetyt muistit vanhenevat tietueen syntymisestä; luku ei virkistä eikä tuhoa tietuetta. Onnistuneet luvut säilyvät ihanteellisessa ulkoisessa lokissa. Lukuvirheet ovat riippumattomia. Yhteinen häviäminen koskee vain viivästettyjä muisteja, ei kaikkia tapahtuman jälkiä. Nämä säädöt eivät muuta aaltofunktiota tai alkuperäistä osumajakaumaa.',
  memoryTimeAxis:'tietueen synty / T₀', memoryCountAxis:'tietueiden odotusarvo / luokka',
});
Object.assign(state, {referenceCopies:1, delayedCopies:1, memoryReads:1,
  memorySpacing:1, memoryReadEfficiency:.9, referenceSurvival:.995,
  memoryLossMode:'independent', memoryPreparations:1000});
let lastMemoryBanks=null;

function evaluateMemoryBanks(r) {
  const d=DATA.historyDynamics,{sig}=current(),eta=state.memoryReadEfficiency;
  const first=d.duration+state.readoutWait*sig, last=first+(state.memoryReads-1)*state.memorySpacing*sig;
  const ref=1-Math.pow(1-state.referenceSurvival,state.referenceCopies), n=state.delayedCopies;
  const reference=[],delayed=[],anyRead=[],either=[],both=[];
  let originalMass=0,unrecovered=0,copyCount=0;
  for(let i=0;i<d.times.length;i++) {
    let firstSuccess=0,R=0;
    // One persistent lifetime per copy; a later read never redraws survival.
    const readSuccess=state.memoryLossMode==='shared'?1-Math.pow(1-eta,n):eta;
    for(let k=0;k<state.memoryReads;k++) {
      const age=(first+k*state.memorySpacing*sig-d.times[i])/sig;
      R=Math.exp(-Math.pow(Math.max(age-state.historyKeep,0)/state.historyFade,state.historyBeta));
      firstSuccess+=R*readSuccess*Math.pow(1-readSuccess,k);
    }
    const pLast=n===0?0:state.memoryLossMode==='shared'?R*readSuccess:1-Math.pow(1-eta*R,n);
    const pAny=n===0?0:state.memoryLossMode==='shared'?firstSuccess:1-Math.pow(1-firstSuccess,n);
    const row=r.before[i];
    reference.push(row.map(p=>p*ref)); delayed.push(row.map(p=>p*pLast));
    anyRead.push(row.map(p=>p*pAny)); both.push(row.map(p=>p*ref*pAny));
    either.push(row.map(p=>p*(ref+(1-ref)*pAny)));
    for(const p of row) {
      originalMass+=p;unrecovered+=p*(1-ref)*(1-pAny);
      copyCount+=p*(state.referenceCopies*state.referenceSurvival+n*eta*R);
    }
  }
  const sum=a=>a.reduce((s,row)=>s+row.reduce((v,p)=>v+p,0),0);
  return {reference,delayed,anyRead,either,both,originalMass,unrecovered,
    missing:1-originalMass,copyCount,first,last,
    probabilities:[sum(both),sum(reference)-sum(both),sum(anyRead)-sum(both),unrecovered,1-originalMass]};
}

function drawMemoryBanks(r=lastDynamics) {
  const m=evaluateMemoryBanks(r);lastMemoryBanks=m;
  const N=state.memoryPreparations,sum=a=>a.flat().reduce((s,p)=>s+p,0);
  for(const id of ['referenceCopies','delayedCopies','memoryReads','memoryPreparations'])$(id+'Value').textContent=state[id];
  $('memorySpacingValue').textContent=state.memorySpacing.toFixed(2);
  $('memoryReadEfficiencyValue').textContent=(100*state.memoryReadEfficiency).toFixed(0)+' %';
  $('referenceSurvivalValue').textContent=(100*state.referenceSurvival).toFixed(1)+' %';
  $('memoryCreated').textContent=(N*m.originalMass).toFixed(1);
  $('memoryLast').textContent=(N*sum(m.delayed)).toFixed(1);
  $('memoryLogged').textContent=(N*sum(m.anyRead)).toFixed(1);
  $('memoryRecovered').textContent=(N*sum(m.either)).toFixed(1);
  $('memorySchedule').textContent=`${t('memoryFirst')}: ${m.first.toFixed(3)} T₀ · ${t('memoryFinal')}: ${m.last.toFixed(3)} T₀. ${t('memoryFromControl')}`;
  $('memoryAccounting').textContent=`${t('memoryUnavailable')}: ${(N*m.unrecovered).toFixed(1)} · ${t('memoryAbsent')}: ${(N*m.missing).toFixed(1)} · ${t('memoryCopiesRead')}: ${(N*m.copyCount).toFixed(1)}.`;
  const {c,w,h}=setupCanvas($('memoryChart')),left=48,right=w-16,top=25,bottom=h-34;
  const curves=[[r.before,'#8fa7b6',[]],[m.reference,'#63a6ff',[5,4]],
    [m.delayed,'#ff6f9f',[]],[m.anyRead,'#ffcb6b',[3,3]],[m.either,'#52e6d8',[]]];
  const totals=curves.map(([joint])=>joint.map(row=>N*row.reduce((s,v)=>s+v,0)));
  const max=Math.max(1e-12,...totals.flat());
  c.clearRect(0,0,w,h);drawAxes(c,w,h,t('memoryTimeAxis'),t('memoryCountAxis'));
  curves.forEach(([,color,dash],k)=>{
    c.beginPath();c.setLineDash(dash);
    totals[k].forEach((p,i)=>{
      const x=left+DATA.historyDynamics.times[i]/DATA.historyDynamics.duration*(right-left),y=bottom-p/max*(bottom-top);
      i?c.lineTo(x,y):c.moveTo(x,y);
    });c.strokeStyle=color;c.lineWidth=2;c.stroke();
  });c.setLineDash([]);c.fillStyle='#8fa7b6';c.fillText(max.toFixed(2),left+4,top);
  for(let time=0;time<=2;time+=.5)c.fillText(String(time),left+time/2*(right-left),bottom+14);
}
for(const id of ['referenceCopies','delayedCopies','memoryReads','memorySpacing',
    'memoryReadEfficiency','referenceSurvival','memoryPreparations','memoryLossMode']) {
  $(id).addEventListener('input',e=>{state[id]=id==='memoryLossMode'?e.target.value:Number(e.target.value);drawMemoryBanks()});
}
