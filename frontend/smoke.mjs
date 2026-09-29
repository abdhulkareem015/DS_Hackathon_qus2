import { chromium } from '@playwright/test';
const browser = await chromium.launch({channel:'msedge',headless:true});
const page = await browser.newPage({viewport:{width:1440,height:1000}});
const errors=[];
page.on('pageerror',e=>errors.push(e.message));
await page.goto('http://127.0.0.1:5173');
await page.getByText('PERFORMANCE SNAPSHOT').waitFor();
for (const name of ['Upload & Data Quality','Maintenance Analysis','Machine Priorities','Decision Tree Model','Prediction','Insights & Action Plan']) {
  await page.getByRole('button',{name,exact:true}).click();
  await page.getByRole('heading',{name,exact:true,level:1}).waitFor();
  if(name==='Maintenance Analysis') {
    await page.locator('.recharts-surface').first().waitFor();
    await page.screenshot({path:'../artifacts/maintenance-analysis.png',fullPage:true});
  }
  if(name==='Prediction') {
    const inputs=page.locator('form input');
    for(let i=0;i<await inputs.count();i++)await inputs.nth(i).fill(i===0?'CNC':'2');
    await page.getByRole('button',{name:'Estimate downtime',exact:true}).click();
    await page.locator('.prediction strong').waitFor();
  }
}
await page.getByRole('button',{name:'Overview',exact:true}).click();
const download=page.waitForEvent('download');
await page.getByRole('link',{name:'Export report'}).click();
await (await download).saveAs('../artifacts/maintenance-report.html');
await page.screenshot({path:'../artifacts/overview-desktop.png',fullPage:true});
await page.setViewportSize({width:390,height:844});
await page.screenshot({path:'../artifacts/overview-mobile.png',fullPage:true});
if(await page.evaluate(()=>document.documentElement.scrollWidth>window.innerWidth))errors.push('Mobile horizontal overflow');
await browser.close();
if(errors.length)throw new Error(errors.join('\n'));
console.log('Browser smoke passed: seven pages, charts, prediction, report download, mobile layout; no page errors.');
