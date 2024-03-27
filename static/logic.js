
  // Variable to hold the interval ID
  var refreshIntervalID;

  // Function to update Bitcoin value and last refresh time
  function updateBTCValue() {
    // Fetch the value from the server
    fetch('/get_btc_value')  // Assuming your Flask route is '/get_btc_value'
      .then(response => response.text())
      .then(data => {
        // Update the HTML content with the fetched value
        document.getElementById('btc_value').textContent = data;
      });

    // Update last refresh time
    var lastRefresh = new Date().getTime();
    document.getElementById('last_refresh').textContent = 0; // Reset to 0 before calculating

    // Clear the previous interval if it exists
    clearInterval(refreshIntervalID);

    // Set a new interval to update the last refresh time
    refreshIntervalID = setInterval(function () {
      var now = new Date().getTime();
      var secondsPassed = Math.floor((now - lastRefresh) / 1000);
      document.getElementById('last_refresh').textContent = secondsPassed;
    }, 1000); // Update every second
  }

  // Call updateBTCValue initially and then every 10 seconds
  updateBTCValue();
  setInterval(updateBTCValue, 10000); // 10000 milliseconds = 10 seconds




  const xValues = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000];
  const name = ['Bitcoin', 'Bot balance'];

  new Chart("myChart", {
    type: "line",
    data: {
      labels: xValues,
      datasets: [{
        data: [860, 1140, 1060, 1060, 1070, 1110, 1330, 2210, 7830, 2478],
        borderColor: "#da9940",
        fill: false,
        label: name[0]
      }, {
        data: [1600, 1700, 1700, 1900, 2000, 2700, 4000, 5000, 6000, 7000],
        borderColor: "lime",
        fill: false,
        label: name[1]
      }]
    },
    options: {
      legend: { display: false }
    }
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
