
let refreshTime = 10;
let btcVal = [];

let timePassedInterval;
let startTime = Date.now();
//displayTimePassed();


(function SiteRefresh() {
  loadData();

  fetch('/get_last_hour_values')
    .then(response => response.json())
    .then(data => {
      btcVal = Object.values(data)[0];

      // Current btc price
      document.getElementById('btc_value').textContent = btcVal[btcVal.length - 1];

      // Last hour diff
      const btcHourDiff = (100 - ((100 / btcVal[btcVal.length - 1]) * btcVal[0])).toFixed(4);
      if (btcHourDiff >= 0) {
        document.getElementById('btcHourDiff').textContent = " +" + btcHourDiff;
      } else {
        document.getElementById('btcHourDiff').textContent = btcHourDiff;
      }

      // Chart
      updateChart();


      resetTimePassedInterval();
      startTime = Date.now(); // Reset the start time

      setTimeout(SiteRefresh, (refreshTime * 1000));
    });
})();


function displayTimePassed() {
  const currentTime = Date.now();
  const timePassedInSeconds = Math.floor((currentTime - startTime) / 1000);
  document.getElementById('last_refresh').textContent = timePassedInSeconds;
}

function resetTimePassedInterval() {
  clearInterval(timePassedInterval); // Clear the previous interval
  timePassedInterval = setInterval(displayTimePassed, 1000); // Update time every second
}





document.getElementById('since_close').textContent = btcVal[btcVal.length - 1];


//Graph
function updateChart() {
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
      legend: { display: false },
      animation: { duration: 0 },
      responsive: true
    }
  });
}

let isTrading = false;

function toggleTrading() {
  if (isTrading) {
    stopTrading();
    document.getElementById('toggleButton').innerText = 'Press to start trading';
    isTrading = false;
  }

  else {
    startTrading();
    document.getElementById('toggleButton').innerText = 'Press to stop trading';
    isTrading = true;
  }
}


function startTrading() {
  fetch('/start_trading', {
    method: 'POST',
  }).then(response => {
    if (response.ok) {
      console.log('Trading started');
    } else {
      console.error('Failed to start trading');
    }
  });

  loadData();
  //startCounting();
}


function stopTrading() {
  fetch('/stop_trading', {
    method: 'POST',
  }).then(response => {
    if (response.ok) {
      console.log('Trading stopped');
    } else {
      console.error('Failed to stop trading');
    }
  });


  clearInterval(intervalId); // Stop the counting interval before saving
  //saveData();
}

function loadData() {
  // Fetch the trading data from the server
  fetch('/load_trading_data', {
    method: 'GET',
    headers: {
      'Content-Type': 'application/json'
    }
  })
    .then(response => {
      if (!response.ok) {
        throw new Error('Network response was not ok');
      }
      return response.json();
    })
    .then(data => {
      if (data.trading_data) {
        document.getElementById('BTC_balance').textContent = data.trading_data['BTC_balance'];
        document.getElementById('USD_balance').textContent = data.trading_data['USD_balance'];

        // Convert the time spent trading from seconds to HH:MM:SS format
        document.getElementById('timeSpentTrading').innerText = formatTime(data.trading_data['time_spent_trading']);

        document.getElementById('money_made').innerText = (data.trading_data['total_profit'] - data.trading_data['total_loss']);

        document.getElementById('good_trades').textContent = data.trading_data['good_trade_count'];
        document.getElementById('bad_trades').textContent = data.trading_data['bad_trade_count'];

        document.getElementById('long_count').textContent = data.trading_data['long_count'];
        document.getElementById('short_count').textContent = data.trading_data['short_count'];

        document.getElementById('leverage').textContent = data.trading_data['leverage'];

      } else {
        console.error('Error: Missing trading data in response:', data);
      }
    })
    .catch(error => {
      console.error('Error fetching or processing trading data:', error);
    });
}

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


/*
function saveData() {
  fetch('/load_trading_data')
    .then(response => response.json())
    .then(data => {
      var tradingData = {};
      tradingData['time_spent_trading'] = formatTime(timeSpent);

      // Update the specific key-value pair you want to update in the trading data
      // For example, let's update the 'time_spent_trading' key
      fetch('/update_trading_data', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ key: 'time_spent_trading', value: tradingData['time_spent_trading'] })
      })
        .then(response => response.json())
        .then(data => console.log(data.message))
        .catch(error => console.error('Error:', error));
    })
    .catch(error => console.error('Error:', error));
}*/

function resetSavedData() {
  fetch('/reset_saved_data', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    }
  })
    .then(response => response.json())
    .then(data => console.log(data.message))
    .catch(error => console.error('Error:', error));
  timeSpent = 0
}


function sliderUpdate(value) {
  document.getElementById('sliderUpdate').textContent = value;
  refreshTime = value;
}

function sliderLeverageUpdate(value) {
  document.getElementById('leverage').textContent = value;
  
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
  document.documentElement.style.setProperty('--darkBlue', '#010207')
  document.documentElement.style.setProperty('--darkerBlue', '#000000bd')
  document.documentElement.style.setProperty('--yellow', '#fce6bd',)
  document.documentElement.style.setProperty('--orange', '#fcdb9f',)
  document.documentElement.style.setProperty('--darkCyan','#04090f',)
  document.documentElement.style.setProperty('--blobColor', '#e8be87')
  document.getElementsByClassName("icon")[0].src = "static/img/Logo.png";
  
}

function lightMode() {
  document.documentElement.style.setProperty('--darkBlue', '#fcfcfc',)
  document.documentElement.style.setProperty('--darkerBlue', '#f7f7f7bd',)
  document.documentElement.style.setProperty('--yellow', '#000000',)
  document.documentElement.style.setProperty('--orange', '#000000',)
  document.documentElement.style.setProperty('--darkCyan','#d3d3d3',)
  document.documentElement.style.setProperty('--blobColor', '#d3d3d3' )
  document.getElementsByClassName("icon")[0].src = "static/img/White_logo.png";
}

function defaultMode() {
  document.documentElement.style.setProperty('--darkBlue', '#041c32',)
  document.documentElement.style.setProperty('--darkerBlue', '#021a31bd',)
  document.documentElement.style.setProperty('--yellow', '#ecb365',)
  document.documentElement.style.setProperty('--orange', '#da9940',)
  document.documentElement.style.setProperty('--darkCyan','#04293a',)
  document.documentElement.style.setProperty('--blobColor', '#06314576' )
  document.getElementsByClassName("icon")[0].src = "static/img/Logo.png";
}







