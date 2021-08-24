$(document).on('submit', '#search', function (e) {
    console.log('hello');
    e.preventDefault();
    $.ajax({
        type: 'POST',
        url: '/',
        data: {
            todo: $("#search").val()
        },
        success: function () {
            alert('saved');
        }
    })
});
