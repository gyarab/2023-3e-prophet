let isTrading;

let leverage = 1;
let commission_rate = 0;

let refreshTime = 10;
let btcVal = [];
let historyVal = [];
let holdVal = [];

let timePassedInterval;
let startTime = Date.now();
let myChart;

let currentLanguage = document.documentElement.lang;


(function AfterLoad() {
  history_array();

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
        isTrading = data.trading_data['is_trading'];

        displayToggleBtn();
        displayBtcBtn();

        document.getElementById('leverage').textContent = data.trading_data['leverage'];
        document.getElementById('commission_rate').textContent = (data.trading_data['commission_rate']);
      } else {
        console.error('Error: Missing trading data in response:', data);
      }
    })
    .catch(error => {
      console.error('Error fetching or processing trading data:', error);

    });



  setTimeout(function () {

    const btcDataset = myChart.data.datasets[0];
    const historyDataset = myChart.data.datasets[1];
    const holdDataset = myChart.data.datasets[2];


    btcDataset.hidden = true;
    historyDataset.hidden = false;
    holdDataset.hidden = false;
  }, 3000);

})();



(function SiteRefresh() {
  history_array();
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

function history_array() {
  fetch('/prepare_array')
    .then(response => response.json())
    .then(data => {
      historyVal = data.history_values;
    })
    .catch(error => console.error('Error fetching prepared array:', error))

  fetch('/prepare_hold_array')
    .then(response => response.json())
    .then(data => {
      holdVal = data.hold_values;
    })
    .catch(error => console.error('Error fetching prepared array:', error))



}



//Graph
function updateChart() {
  const xValues = Array.from({ length: 60 }, (_, i) => 60 - i); // Creating an array from 60 to 1
  let name = [];
  if (currentLanguage == "cz") { name = ['Cena Bitcoinu', 'Celkové finance', 'Nákup & držení'] }
  else name = ['Bitcoin price', 'Total balance', 'Buy & Hold'];

  if (!myChart) {
    // If the chart doesn't exist, create a new one
    myChart = new Chart("myChart", {
      type: "line",
      data: {
        labels: xValues,
        datasets: [{
          data: btcVal,
          borderColor: getComputedStyle(document.documentElement).getPropertyValue('--orange'),
          fill: false,
          label: name[0]
        },
        {
          data: historyVal,
          borderColor: getComputedStyle(document.documentElement).getPropertyValue('--yellow'),
          fill: false,
          label: name[1]
        },
        {
          data: holdVal,
          borderColor: getComputedStyle(document.documentElement).getPropertyValue('--orange'),
          fill: false,
          label: name[2]
        }]
      },
      options: {
        legend: { display: true, onClick: null },
        animation: { duration: 0 },
        responsive: true,
        elements: {
          point: {
            radius: 1.5
          }
        }
      }
    });
    myChart.data.datasets[0].hidden = true;
    updateChart();
  } else {
    // If the chart exists, update its data
    //myChart.data.labels = xValues;
    myChart.data.datasets[0].data = btcVal;
    myChart.data.datasets[0].borderColor = getComputedStyle(document.documentElement).getPropertyValue('--orange');
    myChart.data.datasets[0].label = name[0];
    myChart.data.datasets[1].data = historyVal;
    myChart.data.datasets[1].borderColor = getComputedStyle(document.documentElement).getPropertyValue('--yellow');
    myChart.data.datasets[1].label = name[1];
    myChart.data.datasets[2].data = holdVal;
    myChart.data.datasets[2].borderColor = getComputedStyle(document.documentElement).getPropertyValue('--orange');
    myChart.data.datasets[2].label = name[2];

    myChart.update();
  }
}

function displayBtcBtn() {
  if (!hideBtc) {
    fetch('/static/translations.json')
      .then(response => response.json())
      .then(translations => {
        // Get the translation for the selected language and key 'tmoney'
        const translation = translations[currentLanguage.toLowerCase()].tShowBtc;
        // Set the inner text of the toggleButton element
        document.getElementById('toggleButtonBTC').innerText = translation;
      })
      .catch(error => console.error('Error fetching translations:', error));
  }
  else {
    fetch('/static/translations.json')
      .then(response => response.json())
      .then(translations => {
        // Get the translation for the selected language and key 'tmoney'
        const translation = translations[currentLanguage.toLowerCase()].tShowBot;
        // Set the inner text of the toggleButton element
        document.getElementById('toggleButtonBTC').innerText = translation;
      })
      .catch(error => console.error('Error fetching translations:', error));
  }
}

let hideBtc = false;
function toggleBtcVal() {
  const btcDataset = myChart.data.datasets[0];
  const historyDataset = myChart.data.datasets[1];
  const holdDataset = myChart.data.datasets[2];
  
  if (hideBtc) {
    
    btcDataset.hidden = hideBtc;
    historyDataset.hidden = !hideBtc;
    holdDataset.hidden = !hideBtc;
  }
  else {
    btcDataset.hidden = hideBtc;
    historyDataset.hidden = !hideBtc;
    holdDataset.hidden = !hideBtc;
  }
  hideBtc = !hideBtc;
  displayBtcBtn();
  
  myChart.update();
}



function displayToggleBtn() {

  if (isTrading) {
    fetch('/static/translations.json')
      .then(response => response.json())
      .then(translations => {
        // Get the translation for the selected language and key 'tmoney'
        const translation = translations[currentLanguage.toLowerCase()].tStopTrade;
        // Set the inner text of the toggleButton element
        document.getElementById('toggleButton').innerText = translation;
      })
      .catch(error => console.error('Error fetching translations:', error));
  }
  else {
    fetch('/static/translations.json')
      .then(response => response.json())
      .then(translations => {
        // Get the translation for the selected language and key 'tmoney'
        const translation = translations[currentLanguage.toLowerCase()].tStartTrade;
        // Set the inner text of the toggleButton element
        document.getElementById('toggleButton').innerText = translation;
      })
      .catch(error => console.error('Error fetching translations:', error));



  }
  document.getElementById("reset_btn").disabled = isTrading
  document.getElementById("CommissionSlider").disabled = isTrading;
  document.getElementById("LeverageSlider").disabled = isTrading;
}

function toggleTrading() {
  waitForLoad();
  if (isTrading) {
    stopTrading();
    isTrading = false;
  }
  else {
    saveData(leverage, commission_rate);
    startTrading();
    isTrading = true;
  }
  displayToggleBtn();
}


function startTrading() {
  fetch('/start_trading', {
    method: 'POST',
  }).then(response => {
    if (!response.ok) {
      console.error('Failed to start trading');
    }
  });

  loadData();
  //startCounting();
}


function waitForLoad() {
  document.getElementById("toggleButton").disabled = true;
  setTimeout(function () {
    document.getElementById("toggleButton").disabled = false;
  }, 5000);
}


function stopTrading() {
  fetch('/stop_trading', {
    method: 'POST',
  }).then(response => {
    if (!response.ok) {
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
        document.getElementById('USD_balance').textContent = betterRounding(data.trading_data['USD_balance'], 2);
        document.getElementById('BTC_balance').textContent = betterRounding((data.trading_data['BTC_balance']), 6);


        // Convert the time spent trading from seconds to HH:MM:SS format
        document.getElementById('timeSpentTrading').innerText = formatTime(data.trading_data['time_spent_trading']);


        document.getElementById('money_made').innerText = betterRounding((data.trading_data['total_profit'] - data.trading_data['total_loss']), 3);

        document.getElementById('good_trades').textContent = data.trading_data['good_trade_count'];
        document.getElementById('bad_trades').textContent = data.trading_data['bad_trade_count'];

        document.getElementById('long_count').textContent = data.trading_data['long_count'];
        document.getElementById('short_count').textContent = data.trading_data['short_count'];
        document.getElementById('hold_count').textContent = data.trading_data['hold_count'];


        document.getElementById('winrate').textContent = betterRounding(data.trading_data['good_trade_count'] / (data.trading_data['hold_count'] + data.trading_data['short_count'] + data.trading_data['long_count']) * 100, 2);
        document.getElementById('win-loss').textContent = betterRounding((data.trading_data['total_profit'] / data.trading_data['good_trade_count']) / (data.trading_data['total_loss'] / data.trading_data['bad_trade_count']) * 100, 4);

      } else {
        console.error('Error: Missing trading data in response:', data);
      }
    })
    .catch(error => {
      console.error('Error fetching or processing trading data:', error);
    });
}

function betterRounding(num, decimals) {
  return Math.round((num + Number.EPSILON) * (10 ** decimals)) / (10 ** decimals)
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

function saveData(leverage, commissionRate) {
  const leverageData = {
    key: 'leverage',
    value: parseFloat(leverage)
  };

  const commissionRateData = {
    key: 'commission_rate',
    value: parseFloat(commissionRate)
  };

  // Send both requests concurrently
  Promise.all([
    fetch('/update_trading_data', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(leverageData)
    }),
    fetch('/update_trading_data', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(commissionRateData)
    })
  ])
    .then(responses => {
      // Check if any response is not okay
      const hasError = responses.some(response => !response.ok);
      if (hasError) {
        throw new Error('Network response was not ok');
      }
      // Parse response JSON
      return Promise.all(responses.map(response => response.json()));
    })
    .then(data => {
      // Log success message for both requests
      //console.log('Leverage:', data[0].message);
      //console.log('Commission Rate:', data[1].message);
    })
    .catch(error => console.error('Error:', error));
}


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


function sliderCommissionUpdate(value) {
  document.getElementById('commission_rate').textContent = value;
  commission_rate = value;
}


function sliderLeverageUpdate(value) {
  document.getElementById('leverage').textContent = value;
  leverage = value;
}

function sliderUpdate(value) {
  document.getElementById('sliderUpdate').textContent = value;
  refreshTime = value;
}


//Language
function switchLanguage() {
  var currentLang = document.documentElement.lang;
  var targetLang = currentLang === "en" ? "cz" : "en";
  currentLanguage = targetLang;

  displayToggleBtn();
  displayBtcBtn();
  updateChart();

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
    })
    .catch(error => console.error('Error fetching translations:', error));
}




function darkMode() {
  document.documentElement.style.setProperty('--darkBlue', '#010207')
  document.documentElement.style.setProperty('--darkerBlue', '#000000bd')
  document.documentElement.style.setProperty('--yellow', '#fce6bd',)
  document.documentElement.style.setProperty('--orange', '#feb025',)
  document.documentElement.style.setProperty('--darkCyan', '#04090f',)
  document.documentElement.style.setProperty('--blobColor', '#04081a')
  document.getElementsByClassName("icon")[0].src = "static/img/Logo.png";
  updateChart();
}

function lightMode() {
  document.documentElement.style.setProperty('--darkBlue', '#fcfcfc',)
  document.documentElement.style.setProperty('--darkerBlue', '#f7f7f7bd',)
  document.documentElement.style.setProperty('--yellow', '#676767',)
  document.documentElement.style.setProperty('--orange', '#000000',)
  document.documentElement.style.setProperty('--darkCyan', '#d3d3d3',)
  document.documentElement.style.setProperty('--blobColor', '#d3d3d3')
  document.getElementsByClassName("icon")[0].src = "static/img/White_logo.png";
  updateChart();
}

function defaultMode() {
  document.documentElement.style.setProperty('--darkBlue', '#041c32',)
  document.documentElement.style.setProperty('--darkerBlue', '#021a31bd',)
  document.documentElement.style.setProperty('--yellow', '#ecb365',)
  document.documentElement.style.setProperty('--orange', '#da9940',)
  document.documentElement.style.setProperty('--darkCyan', '#04293a',)
  document.documentElement.style.setProperty('--blobColor', '#06314576')
  document.getElementsByClassName("icon")[0].src = "static/img/Logo.png";
  updateChart();
}







