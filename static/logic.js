


const refreshTime = 10;
let btcVal = [];

(function SiteRefresh()
{
  fetch('/get_last_hour_values')
  .then(response => response.json())
  .then(data => {
    btcVal = Object.values(data)[0]
  

    //Current btc price
    document.getElementById('btc_value').textContent = btcVal[btcVal.length - 1];

    // Last hour diff
    const btcHourDiff = (100 - ((100 / btcVal[btcVal.length-1]) * btcVal[0])).toFixed(4)
    if(btcHourDiff >= 0){document.getElementById('btcHourDiff').textContent = " +"+btcHourDiff;}
    else document.getElementById('btcHourDiff').textContent = btcHourDiff;

    // Chart
    updateChart();
    
    
  
    console.log(btcVal)
  setTimeout(SiteRefresh, (refreshTime * 1000))
  });
})//()




//Graph
function updateChart(){
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
}


function loadData() {
  // Fetch the trading data from the server
  fetch('/load_trading_data', {
      method: 'POST',
      headers: {
          'Content-Type': 'application/json'
      }
  })
  .then(response => response.json())
  .then(data => {
      if (data.trading_data) {
          // Update the content of the spans with the trading data
          document.getElementById('btc_value_now').textContent = data.trading_data['Btc value at close app'];
          document.getElementById('USD_in_wallet').textContent = data.trading_data['USD in wallet'];
          
          
          document.getElementById('timeSpentTrading').innerText = "Time Spent Trading: " + data.trading_data['time_spent_trading'];
          
          startCounting();
          
      } else {
          console.error('Error loading trading data:', data.message);
      }
  })
  .catch(error => {
      console.error('Error fetching trading data:', error);
  });
}



//save and load data
let intervalId; // Variable to store the interval ID
let timeSpent = 0; // Variable to store the time spent trading

function startCounting() {
    intervalId = setInterval(() => {
        timeSpent++; // Increase the time spent trading by 1 second
        document.getElementById('timeSpentTrading').innerText = formatTime(timeSpent);
    }, 1000);
}

// Format time in HH:MM:SS format
function formatTime(seconds) {
    var hours = Math.floor(seconds / 3600);
    var minutes = Math.floor((seconds % 3600) / 60);
    var remainingSeconds = seconds % 60;
    return pad(hours) + ":" + pad(minutes) + ":" + pad(remainingSeconds);
}

// Add leading zero if needed
function pad(value) {
    return value < 10 ? "0" + value : value;
}








//Language
function switchLanguage() {
  var currentLang = document.documentElement.lang;
  var targetLang = currentLang === "en" ? "cz" : "en";
  
  fetch('/static/translations.json')
      .then(response => response.json())
      .then(data => {
        // Get all elements with data-translation attribute
        var elements = document.querySelectorAll('[data-translation]');

        // Update content of each element with the corresponding translation
        elements.forEach(element => {
          var translationKey = element.getAttribute('data-translation');
          element.textContent = data[targetLang][translationKey];
        });

        // Set the document language to the target language
        document.documentElement.lang = targetLang;
        console.log(targetLang);
      })
      .catch(error => console.error('Error fetching translations:', error));
}




function darkMode() {
  document.documentElement.style.setProperty('--darkBlue', '#010207' )
  document.documentElement.style.setProperty('--darkerBlue', '#000000')
  document.documentElement.style.setProperty('--yellow', '#fce6bd',)
  document.documentElement.style.setProperty('--orange', '#fcdb9f',)
  document.documentElement.style.setProperty('--darkCyan','#04090f',) 
}

function lightMode() {
  document.documentElement.style.setProperty('--darkBlue', '#fcfcfc', )
  document.documentElement.style.setProperty('--darkerBlue', '#f7f7f7',)
  document.documentElement.style.setProperty('--yellow', '#000000',)
  document.documentElement.style.setProperty('--orange', '#000000',)
  document.documentElement.style.setProperty('--darkCyan','#d3d3d3',)
}

function defaultMode() {
  document.documentElement.style.setProperty('--darkBlue', '#041c32', )
  document.documentElement.style.setProperty('--darkerBlue', '#021a31',)
  document.documentElement.style.setProperty('--yellow', '#ecb365',)
  document.documentElement.style.setProperty('--orange', '#da9940',)
  document.documentElement.style.setProperty('--darkCyan','#04293a',)
}





function saveData() {
  clearInterval(intervalId); // Stop the counting interval before saving
  fetch('/load_trading_data')
  .then(response => response.json())
  .then(data => {
      var tradingData = {};
      //tradingData['time_spent_trading'] = formatTime(timeSpent); // Update the time spent trading in the trading data
      tradingData['btc_value_invested'] = data.trading_data['BTC Value Invested']; // Get the BTC value from the loaded data
      fetch('/save_trading_data', {
          method: 'POST',
          headers: {
              'Content-Type': 'application/json'
          },
          body: JSON.stringify(tradingData)
      })
      .then(response => response.json())
      .then(data => console.log(data.message))
      .catch(error => console.error('Error:', error));
  })
  .catch(error => console.error('Error:', error));
}




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
          //document.getElementById('btc_value').textContent = data;

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
setInterval(updateBTCValue, 1000000); // 10000 milliseconds = 10 seconds
