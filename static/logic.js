
  // Variable to hold the interval ID
  var refreshIntervalID;

  // Function to update Bitcoin value and last refresh time
function updateBTCValue() {
  // Fetch the value from the server
  fetch('/get_btc_value')  
      .then(response => {
          if (!response.ok) {
              throw new Error('Network response was not ok');
          }
          return response.text();
      })
      .then(data => {
          // Update the HTML content with the fetched value
          document.getElementById('btc_value').textContent = data;

          // Update last refresh time
          var lastRefresh = new Date().getTime();
          document.getElementById('last_refresh').textContent = 0; 

          // Clear the previous interval if it exists
          clearInterval(refreshIntervalID);

          // Set a new interval to update the last refresh time
          refreshIntervalID = setInterval(function () {
              var now = new Date().getTime();
              var secondsPassed = Math.floor((now - lastRefresh) / 1000);
              document.getElementById('last_refresh').textContent = secondsPassed;
          }, 1000); // Update every second
      })
      .catch(error => {
          console.error('There was a problem with the fetch operation:', error);

          // Increment last refresh time until response is received
          var secondsPassed = parseInt(document.getElementById('last_refresh').textContent || 0, 10);
          document.getElementById('last_refresh').textContent = secondsPassed + 1;

          // Retry after 1 second
          setTimeout(updateBTCValue, 1000);
      });
}

// Call updateBTCValue initially and then every 10 seconds
updateBTCValue();
setInterval(updateBTCValue, 10000); // 10000 milliseconds = 10 seconds



function refresh()
{
  
}



//Graph
fetch('/get_last_hour_values')
  .then(response => response.json())
  .then(data => {
    const btcVal = Object.values(data)[0]


    const btcHourDiff = (100 - ((100 / btcVal[btcVal.length-1]) * btcVal[0])).toFixed(4)
    if(btcHourDiff >= 0){document.getElementById('btcHourDiff').textContent = " +"+btcHourDiff;}
    else document.getElementById('btcHourDiff').textContent = btcHourDiff;
    
    

    const xValues = Array.from({ length: 60 }, (_, i) => 60 - i); // Creating an array from 60 to 1
    const name = ['Bitcoin', 'Bot balance'];

    new Chart("myChart", {
      type: "line",
      data: {
        labels: xValues,
        datasets: [{
          data: btcVal, 
          borderColor: "#da9940",
          fill: false,
          label: name[0]
        }, /*{
          data: [1600, 1700, 1700, 1900, 2000, 2700, 4000, 5000, 6000, 7000],
          borderColor: "lime",
          fill: false,
          label: name[1]
        }*/]
      },
      options: {
        legend: { display: false }
      }
    });
  });



function darkMode() {
  document.documentElement.style.setProperty('--darkBlue', '#010207' );
  document.documentElement.style.setProperty( '--darkerBlue', '#000000')
  document.documentElement.style.setProperty(  '--yellow', '#fce6bd',)
  document.documentElement.style.setProperty('--orange', '#fcdb9f',)
  document.documentElement.style.setProperty('--darkCyan','#04090f',) 
    
    
}

function lightMode() {
  document.documentElement.style.setProperty('--darkBlue', '#fcfcfc', );
  document.documentElement.style.setProperty('--darkerBlue', '#f7f7f7',)
  document.documentElement.style.setProperty('--yellow', '#',)
  document.documentElement.style.setProperty('--orange', '#',)
  document.documentElement.style.setProperty( '--darkCyan','#d3d3d3',)
  
}

function defaultMode() {
  document.documentElement.style.setProperty('--darkBlue', '#041c32', );
  document.documentElement.style.setProperty('--darkerBlue', '#021a31',)
  document.documentElement.style.setProperty('--yellow', '#ecb365',)
  document.documentElement.style.setProperty('--orange', '#da9940',)
  document.documentElement.style.setProperty( '--darkCyan','#04293a',)
  
}
